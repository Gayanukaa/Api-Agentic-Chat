class ExplanationAgent:
    def __init__(self, llm):
        """
        Initialize the Explanation Agent with an LLM (Large Language Model).

        Parameters:
        - llm: An object with an `invoke` method for generating text responses.
        """
        self.llm = llm

    def generate_explanation(self, user_query, results):
        """
        Generate a detailed explanation for the user query based on relevant API documentation.

        Parameters:
        - user_query: The query provided by the user.
        - results: A list of relevant API documentation chunks. Each chunk is a dictionary with keys
                   like 'endpoint', 'description', and 'file_name'.

        Returns:
        - A string containing the generated explanation.
        """
        # Combine relevant documents into a context
        context = "\n\n".join([
            f"Endpoint: {doc.get('endpoint', 'N/A')}\n"
            f"Description: {doc.get('description', 'N/A')}\n"
            f"File Name: {doc.get('file_name', 'N/A')}"
            for doc in results
        ])

        # Prepare the prompt
        prompt_template = """
        User Query: {query}

        Relevant API Documentation:
        {context}

        Based on the documentation above, explain the process clearly and step-by-step.
        """
        prompt = prompt_template.format(query=user_query, context=context)

        # Generate the explanation using the LLM
        explanation = self.llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        return explanation


