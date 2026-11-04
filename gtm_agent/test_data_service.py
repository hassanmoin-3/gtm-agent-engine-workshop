import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.object(data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.prospect_id = "LEAD-71001"

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_update_refreshes_cached_profile(self):
        gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(result["found"])
        self.assertIn("Kafka", result["prospect_profile"]["tech_stack"])
        self.assertIn("Kafka", data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]["tech_stack"])

    def test_existing_technology_is_not_duplicated(self):
        result = data_service.update_prospect_info(self.prospect_id, "Snowflake")

        self.assertFalse(result["updated"])
        self.assertTrue(result["found"])
        self.assertEqual(result["tech_stack"].count("Snowflake"), 1)
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Snowflake"), 1)

    def test_score_uses_saved_stack_after_update(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]
        offering = data_service.get_offering("OFFER-10005")

        def score(messages):
            content = messages[-1]["content"]
            saved_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
            self.assertIn("Kafka", saved_profile["tech_stack"])
            return gtm_agent.ProspectScore(
                score=100,
                justification="Kafka is present; all required technologies are present.",
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        update = gtm_agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Kafka"})
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score
            result = gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        self.assertTrue(update["updated"])
        self.assertEqual(result["score"], 100)
        self.assertIn("Kafka is present", result["justification"])

    def test_unknown_prospect_is_not_found(self):
        result = data_service.update_prospect_info("UNKNOWN", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
