import collections.abc
import http.server
import json
import pathlib
import threading
from unittest import mock
import urllib.request

from absl.testing import absltest

from litert_lm_cli import model
from litert_lm_cli import serve


def _parse_sse_events(
    lines: collections.abc.Iterable[str],
) -> list[dict[str, str]]:
  events = []
  current_event = {}
  for line in lines:
    if line.startswith("event: "):
      current_event["event"] = line[len("event: ") :]
    elif line.startswith("data: "):
      current_event["data"] = line[len("data: ") :]
    elif not line and current_event:
      events.append(current_event)
      current_event = {}
  return events


class ServeOpenAIStreamingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(serve, "_current_engine", None))
    self.enter_context(mock.patch.object(serve, "_current_model_id", None))

    self.server = http.server.HTTPServer(("localhost", 0), serve.OpenAIHandler)
    self.port = self.server.server_port

    self.server_thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )
    self.server_thread.start()

    self.model_path = (
        pathlib.Path(absltest.get_default_test_srcdir())
        / "google3/runtime/e2e_tests/data/gemma3-1b-it-int4.litertlm"
    )

  def tearDown(self):
    self.server.shutdown()
    self.server.server_close()
    self.server_thread.join()
    super().tearDown()

  def test_openai_responses_streaming(self):
    self.assertTrue(
        self.model_path.exists(), f"Model not found at {self.model_path}"
    )

    mock_from_id = self.enter_context(
        mock.patch.object(model.Model, "from_model_id", autospec=True)
    )
    mock_from_id.return_value = model.Model(
        model_id="gemma3", model_path=str(self.model_path)
    )

    data = json.dumps(
        {"model": "gemma3", "input": "Say hi", "stream": True}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"http://localhost:{self.port}/v1/responses",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as response:
      self.assertEqual(response.getcode(), 200)
      self.assertEqual(response.getheader("Content-Type"), "text/event-stream")

      lines = response.read().decode("utf-8").split("\n")

      events = _parse_sse_events(lines)

      self.assertNotEmpty(events)

      with self.subTest(name="Verify created event"):
        created_event = events[0]
        self.assertEqual(created_event["event"], "response.created")
        created_data = json.loads(created_event["data"])
        self.assertIn("id", created_data)
        self.assertEqual(created_data["status"], "in_progress")

      with self.subTest(name="Verify delta events"):
        delta_events = [
            e for e in events if e.get("event") == "response.output_text.delta"
        ]
        self.assertNotEmpty(delta_events)
        for de in delta_events:
          delta_data = json.loads(de["data"])
          self.assertIn("delta", delta_data)
          self.assertIn("text", delta_data["delta"])

      with self.subTest(name="Verify completed event"):
        completed_event = next(
            e for e in events if e.get("event") == "response.completed"
        )
        completed_data = json.loads(completed_event["data"])
        self.assertIn("id", completed_data)
        self.assertEqual(completed_data["status"], "completed")

      with self.subTest(name="Verify DONE message"):
        self.assertIn("data: [DONE]", lines)


if __name__ == "__main__":
  absltest.main()
