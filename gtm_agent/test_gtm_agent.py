"""Numbered cases for ID-based email and scoring tools."""

import importlib
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch(
    "langchain_openai.ChatOpenAI", side_effect=lambda **kwargs: ChatOpenAI(api_key="test", **kwargs)
):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "name": "Test Prospect", "email": "prospect@example.com",
            "engagement_history": [], "account_details": [], "tech_stack": ["AWS"],
        }
        self.offering = {
            "description": "Cloud platform", "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1,
        }
        self.rep = {"rep_id": "rep_test", "name": "Test Rep", "email": "rep@example.com"}
        for name, value in (
            ("PROSPECTS", {"LEAD-test": self.prospect}),
            ("OFFERINGS", {"OFFER-test": self.offering}),
            ("REP_IDS", [self.rep]), ("_PROFILES", {}),
        ):
            self.enterContext(patch.object(agent.data_service, name, value))
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_test"}})
        self.scorer = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.score = {"score": 80, "justification": "Good fit"}
        self.scorer.invoke.return_value.model_dump.return_value = self.score

    def test_1_email_resolves_contact_and_sender(self):
        result = agent.send_prospect_email.func("LEAD-test", "Hello", "Demo?", self.runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual((result["to"], result["to_name"]), (self.prospect["email"], self.prospect["name"]))
        self.assertEqual((result["from"], result["from_name"]), (self.rep["email"], self.rep["name"]))
        self.assertEqual((result["subject"], result["body"]), ("Hello", "Demo?"))

    def test_2_missing_email_prospect_does_not_send(self):
        with patch.object(agent.uuid, "uuid4") as message_id:
            result = agent.send_prospect_email.func("missing", "Hello", "Demo?", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect not found."})
        message_id.assert_not_called()

    def test_3_scoring_loads_records_from_ids(self):
        self.assertEqual(agent.score_prospect.invoke({"prospect_id": "LEAD-test", "offering_id": "OFFER-test"}), self.score)
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), self.offering)
        self.assertEqual(json.loads(profile), {"prospect_id": "LEAD-test", **self.prospect})

    def test_4_missing_scoring_prospect(self):
        self.assertEqual(agent.score_prospect.func("missing", "OFFER-test"), {"score": None, "error": "Prospect not found."})
        self.scorer.invoke.assert_not_called()

    def test_5_missing_scoring_offering(self):
        self.assertEqual(agent.score_prospect.func("LEAD-test", "missing"), {"score": None, "error": "Cannot score without a valid offering."})
        self.scorer.invoke.assert_not_called()

    def test_6_refresh_and_explicit_lookups(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-test"})
        self.assertEqual(profile, {"found": True, "prospect_profile": {"prospect_id": "LEAD-test", **self.prospect}})
        self.prospect["tech_stack"] = ["AWS", "Kafka"]
        agent.score_prospect.func("LEAD-test", "OFFER-test")
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        self.assertEqual(json.loads(content.split("Prospect profile:\n")[1])["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": "OFFER-test"}), {"found": True, "offering": self.offering})
        self.assertEqual(agent.lookup_offering.invoke({"offering_id": "missing"}), {"found": False, "offering": None})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {"found": False, "prospect_profile": None})

    def test_7_graph_accepts_direct_email_and_score_calls(self):
        self.assertEqual(agent.get_current_rep.func(self.runtime), {"found": True, "rep": self.rep})
        self.assertNotIn("always call the get_current_rep", agent.SYSTEM_PROMPT)
        for name, args, expected in (
            ("send_prospect_email", {"prospect_id": "LEAD-test", "subject": "Hello", "body": "Demo?"}, "sent"),
            ("score_prospect", {"prospect_id": "LEAD-test", "offering_id": "OFFER-test"}, 80),
        ):
            with self.subTest(tool=name):
                model = ScriptedModel(responses=[
                    AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-test"}]),
                    AIMessage(content="Done"),
                ])
                graph = agent.create_deep_agent(
                    model=model, tools=[agent.send_prospect_email, agent.score_prospect, agent.get_current_rep],
                    system_prompt=agent.SYSTEM_PROMPT,
                )
                result = graph.invoke({"messages": [{"role": "user", "content": "Complete the request"}]}, config=self.runtime.config)
                outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(len(outputs), 1)
                payload = json.loads(outputs[0].content)
                self.assertEqual(payload.get("status", payload.get("score")), expected)


if __name__ == "__main__":
    unittest.main()
