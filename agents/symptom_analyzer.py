"""Symptom Analysis Agent — identifies possible conditions from patient data + RAG context."""

import json
from typing import Dict, Any
from agents.base import BaseAgent


class SymptomAnalysisAgent(BaseAgent):
    name = "SymptomAnalysisAgent"

    system_prompt = """你是一位专业的儿童语言与心理障碍症状分析专家。

你的任务是：
1. 根据患者的基本信息和症状描述，结合知识库检索结果，分析可能的疾病类型
2. 识别关键症状并标注严重程度
3. 给出初步的鉴别分析

输出要求（严格JSON格式）：
{
  "possible_conditions": [
    {
      "name": "疾病名称",
      "confidence": "高/中/低",
      "matching_symptoms": ["匹配的症状1", "匹配的症状2"],
      "unmatched_symptoms": ["未匹配的症状"]
    }
  ],
  "key_symptoms": ["关键症状1", "关键症状2"],
  "analysis_summary": "综合分析说明"
}

注意：
- 最多列出3个可能的疾病，按可能性排序
- confidence基于症状匹配程度判断
- 分析要结合患者年龄和病史
- 不要做出确诊结论，仅提供可能性分析"""

    def run(self, patient_data: Dict[str, Any], rag_context: str) -> Dict[str, Any]:
        user_prompt = f"""请分析以下患者信息：

【患者信息】
年龄: {patient_data.get('age', '未知')}
症状描述: {patient_data.get('symptoms', '未知')}
病史: {patient_data.get('history', '未知')}

【知识库检索结果】
{rag_context}

请输出JSON格式的分析结果。"""

        response = self._call_llm(user_prompt)
        return self._parse_json(response)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "possible_conditions": [],
                "key_symptoms": [],
                "analysis_summary": text,
                "_raw_response": text,
            }
