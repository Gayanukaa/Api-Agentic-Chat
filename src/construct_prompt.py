import json

class ExplanationAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_explanation(self, user_query, results, chat_history):
        # Create a concise context from the selected results
        context = "\n\n".join([ 
            f"Endpoint: {doc.get('endpoint', 'N/A')}\n"
            f"Description: {doc.get('description', 'N/A')}\n"
            f"Body: {json.dumps(doc.get('body', {}), indent=2) if doc.get('body') else 'No body available'}"
            for doc in results
        ])

        # Combine the chat history into the prompt, making sure the history is included
        chat_history_str = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history])

        # Prepare the prompt template
        prompt_template = """
        Chat History:
        {chat_history}

        User Query: {query}

        Relevant API Documentation:
        {context}

        Based on the documentation, explain the process step-by-step, covering all relevant APIs in a concise and user-friendly manner. Avoid repeating unnecessary information.
        """
        
        # Format the prompt with user query, chat history, and context
        prompt = prompt_template.format(query=user_query, context=context, chat_history=chat_history_str)

        # Generate the explanation
        explanation = self.llm.invoke([ 
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

        return explanation
