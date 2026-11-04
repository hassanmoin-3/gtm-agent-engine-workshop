import importlib
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODEL_API_KEY = os.environ.get("OPENAI_API_KEY")
with patch.dict(os.environ, {
    "OPENAI_API_KEY": MODEL_API_KEY or "test-key",
    "LANGSMITH_TRACING": "false",
}), patch("dotenv.load_dotenv"):
    agent_module = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect):
        return agent_module.send_prospect_email.func(
            prospect, "Introduction", "Hello from your sales rep.", self.runtime,
        )

    def test_disqualified_prospect_is_blocked_before_message_generation(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }
        with patch.object(agent_module.uuid, "uuid4") as message_id:
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "prospect_id": "LEAD-50001",
        })
        self.assertNotIn("message_id", result)
        message_id.assert_not_called()

    def test_eligible_prospect_retains_send_behavior(self):
        prospect = agent_module.get_prospect.invoke({"prospect_id": "LEAD-26912"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["to_name"], prospect["name"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["from_name"], "Ava Mills")
        self.assertEqual(result["subject"], "Introduction")
        self.assertEqual(result["body"], "Hello from your sales rep.")

    def test_prospect_without_id_retains_send_behavior(self):
        prospect = {"name": "Test Prospect", "email": "prospect@example.com"}
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertRegex(result["message_id"], r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result["to"], prospect["email"])

    def test_unknown_prospect_id_retains_send_behavior(self):
        result = self.send({"prospect_id": "unknown", "email": "prospect@example.com"})
        self.assertEqual(result["status"], "sent")

    def test_missing_email_retains_failure(self):
        self.assertEqual(self.send({}), {
            "status": "failed", "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(MODEL_API_KEY, "Live agent check requires OPENAI_API_KEY")
    def test_agent_reports_disqualified_email_was_not_sent(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent_module.run_agent(
                "Email LEAD-50001 with subject Introduction and body Hello from your sales rep.",
                user_id="rep_amills",
            )
        sends = [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertTrue(sends)
        self.assertTrue(all(send["status"] == "blocked" for send in sends))
        self.assertTrue(all("message_id" not in send for send in sends))
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not sent|wasn't sent|did not send|didn't send|cannot send|can't send")


if __name__ == "__main__":
    unittest.main()
