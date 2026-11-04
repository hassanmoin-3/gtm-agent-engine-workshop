import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"), \
        patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_contact_allowlist_and_email_flow(self):
        for pid, record in agent.data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                result = agent.get_prospect.invoke({"prospect_id": pid})
                self.assertTrue(result["found"])
                contact = result["prospect"]
                self.assertEqual(contact, {
                    "prospect_id": pid,
                    "name": record["name"],
                    "email": record["email"],
                    "disqualified": record["disqualified"],
                })
                sent = agent.send_prospect_email.func(
                    prospect=contact, subject="Follow up", body="Hello",
                    runtime=SimpleNamespace(config={}),
                    from_rep=agent.data_service.REP_IDS[0],
                )
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["to"], record["email"])
                self.assertEqual(sent["to_name"], record["name"])
                self.assertNotIn("billing_qualification", json.dumps(sent))

    def test_profile_cache_and_scoring_flow(self):
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})["offering"]
        for pid, record in agent.data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                self.assertTrue(result["found"])
                profile = result["prospect_profile"]
                self.assertNotIn("billing_qualification", json.dumps(result))
                for field in ("annual_revenue", "tech_stack", "account_details",
                              "engagement_history"):
                    self.assertEqual(profile[field], record[field])
                self.assertEqual(agent.data_service._PROFILES[pid], profile)
                self.assertEqual(
                    agent.build_prospect_profile.invoke({"prospect_id": pid}), result,
                )
                self.assert_scoring_input(profile, offering)

    def assert_scoring_input(self, profile, offering):
        expected = agent.ProspectScore(
            score=75, justification="Fit based on revenue and technology.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        model = Mock()
        model.invoke.return_value = expected
        with patch.object(agent, "_scoring_llm", model):
            result = agent.score_prospect.invoke({
                "prospect_profile": profile, "offering": offering,
            })
        self.assertEqual(result, expected.model_dump())
        messages = model.invoke.call_args.args[0]
        self.assertNotIn("billing_qualification", json.dumps(messages))
        supplied = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        for field in ("annual_revenue", "tech_stack", "segment", "engagement_history"):
            if field in profile:
                self.assertEqual(supplied[field], profile[field])

    def test_profile_retains_segment(self):
        pid = next(iter(agent.data_service.PROSPECTS))
        record = {**agent.data_service.PROSPECTS[pid], "segment": "Enterprise"}
        with patch.dict(agent.data_service.PROSPECTS, {pid: record}):
            profile = agent.build_prospect_profile.invoke({"prospect_id": pid})["prospect_profile"]
        self.assertEqual(profile["segment"], "Enterprise")
        self.assertNotIn("billing_qualification", profile)

    def test_direct_unfiltered_scoring_input(self):
        profile = {
            "annual_revenue": 70000000, "tech_stack": ["AWS"],
            "segment": "Enterprise", "engagement_history": [{"channel": "Webinar"}],
            "billing_qualification": {"tax_id": "private-test-value"},
        }
        offering = agent.data_service.OFFERINGS["OFFER-10001"]
        self.assert_scoring_input(profile, offering)
        self.assertIn("billing_qualification", profile)

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()
