import json
import os
import unittest
import uuid
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    with patch("dotenv.load_dotenv"):
        from . import gtm_agent


class AgentTests(unittest.TestCase):
    def test_tool_workflow_and_first_call_context(self):
        requests = []
        replies = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "rep", "type": "function", "function": {
                    "name": "get_current_rep", "arguments": "{}"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "offering", "type": "function", "function": {
                    "name": "lookup_offering", "arguments": json.dumps({
                        "offering_id": "OFFER-10001"})}},
            ]},
            {"role": "assistant", "content": "Here is the offering, Ava."},
        ]

        def respond(**kwargs):
            requests.append(kwargs)
            return Mock(parse=Mock(return_value={"id": "test", "model": gtm_agent.MODEL_NAME, "choices": [
                {"index": 0, "finish_reason": "stop", "message": replies.pop(0)},
            ]}))

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            with patch.object(gtm_agent.agent_model.client.with_raw_response, "create", side_effect=respond):
                result = gtm_agent.gtm_agent.invoke(
                    {"messages": [{"role": "user", "content": "Look up OFFER-10001"}]},
                    config={"metadata": {"user_id": "rep_amills"}},
                )

        first = requests[0]
        self.assertEqual(first["messages"], [
            {"role": "system", "content": gtm_agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Look up OFFER-10001"},
        ])
        schemas = {tool["function"]["name"]: tool["function"] for tool in first["tools"]}
        self.assertEqual(set(schemas), {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "get_current_rep", "send_prospect_email", "score_prospect",
            "update_prospect_info",
        })
        self.assertNotIn("runtime", schemas["get_current_rep"]["parameters"]["properties"])
        tool_results = {message.name: json.loads(message.content)
                        for message in result["messages"] if message.type == "tool"}
        self.assertEqual(tool_results["get_current_rep"], {
            "rep": gtm_agent.data_service.get_rep("rep_amills"), "found": True,
        })
        self.assertEqual(tool_results["lookup_offering"], {
            "offering": gtm_agent.data_service.get_offering("OFFER-10001"), "found": True,
        })
        self.assertIsInstance(result["messages"][-1], AIMessage)
        self.assertEqual(result["messages"][-1].content, "Here is the offering, Ava.")
        self.assertEqual(len(requests), 3)

    def test_run_agent_preserves_config_and_result(self):
        messages = [AIMessage(content="The offering is available.")]
        with patch.object(gtm_agent.gtm_agent, "invoke", return_value={"messages": messages}) as invoke:
            result = gtm_agent.run_agent(
                "Look up OFFER-10001", user_id="rep_amills",
                environment="test", thread_id="thread-123",
            )
        config = invoke.call_args.kwargs["config"]
        self.assertEqual(invoke.call_args.args, (
            {"messages": [{"role": "user", "content": "Look up OFFER-10001"}]},
        ))
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertIsInstance(config["run_id"], uuid.UUID)
        self.assertEqual(config["metadata"], {
            "thread_id": "thread-123", "user_id": "rep_amills",
            "environment": "test", "request_intent": "lookup_offering",
        })
        self.assertEqual(result, {
            "reply": messages[-1].content, "messages": messages,
            "run_id": str(config["run_id"]),
        })


if __name__ == "__main__":
    unittest.main()
