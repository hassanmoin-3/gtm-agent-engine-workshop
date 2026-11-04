import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key"), "LANGSMITH_TRACING": "false"}):
    module = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def send(self, body, subject="Pricing discussion"):
        return module.send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject=subject,
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}}),
        )

    def test_attachment_claims_fail(self):
        for body in (
            "I've attached our pricing deck.",
            "I have attached the document.",
            "I’m attaching our pricing deck.",
            "We are attaching the proposal.",
            "Please find attached our pricing deck.",
            "The pricing deck is attached.",
            "Attached is the pricing deck.",
            "See the attached PDF.",
            "Review our enclosed brochure.",
            "The attachment is included.",
            "No brochure is attached, but I've attached the pricing deck.",
        ):
            for variant in (body, body.upper(), body.swapcase()):
                with self.subTest(body=variant):
                    result = self.send(variant)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("cannot include file attachments", result["error"])
                    self.assertNotIn("message_id", result)

    def test_truthful_and_ordinary_emails_are_sent(self):
        for body in (
            "No attachment is included.",
            "No attachments are included.",
            "There is no attachment included.",
            "The pricing deck is not attached.",
            "I haven't attached the deck.",
            "I have not attached any files.",
            "I cannot attach files; the deck was not included.",
            "We can discuss attachment rates next week.",
            "We attached importance to your feedback.",
            "The prospect feels attached to their current provider.",
            "Would you like to discuss pricing next week?",
        ):
            with self.subTest(body=body):
                result = self.send(body)
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["body"], body)
                self.assertEqual(result["to"], "prospect@example.com")
                self.assertTrue(result["message_id"])

    def test_subject_cannot_claim_an_attachment(self):
        self.assertEqual(self.send("Let's discuss pricing.", "Pricing deck attached")["status"], "failed")


@unittest.skipUnless(HAS_MODEL_KEY, "Document-request agent test requires OPENAI_API_KEY")
class DocumentRequestAgentTests(unittest.TestCase):
    def test_document_request_discloses_missing_document(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = module.run_agent(
                "Email LEAD-50003 to send over our pricing deck.", user_id="rep_sbrown"
            )
        sent = []
        for message in result["messages"]:
            if getattr(message, "name", None) == "send_prospect_email":
                payload = json.loads(message.content)
                if payload.get("status") == "sent":
                    sent.append(payload)
        for email in sent:
            self.assertNotRegex(
                email["body"].lower(),
                r"(?:attached is|please find attached|i['’]ve attached|deck is attached)",
            )
        self.assertRegex(result["reply"].lower(), r"(?:deck|document|file|attachment|collateral)")
        self.assertRegex(
            result["reply"].lower(),
            r"(?:cannot|can't|unable|not included|not attached|without.{0,30}(?:deck|attachment)|how.{0,20}proceed|would you|could you|please provide)",
        )


if __name__ == "__main__":
    unittest.main()
