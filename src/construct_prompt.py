import json

class ExplanationAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_explanation(self, user_query, results):
        # Create a concise context from the selected results
        context = "\n\n".join([
            f"Endpoint: {doc.get('endpoint', 'N/A')}\n"
            f"Description: {doc.get('description', 'N/A')}\n"
            f"Body: {json.dumps(doc.get('body', {}), indent=2) if doc.get('body') else 'No body available'}"
            for doc in results
        ])

        # Prepare the prompt
        prompt_template = """
        User Query: {query}

        Relevant API Documentation:
        {context}

        Based on the documentation, explain the process step-by-step, covering all relevant APIs in a concise and user-friendly manner. Avoid repeating unnecessary information.
        """
        prompt = prompt_template.format(query=user_query, context=context)

        # Generate the explanation
        explanation = self.llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        return explanation
