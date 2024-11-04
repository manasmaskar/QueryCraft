from transformers import T5Tokenizer

class CustomT5Tokenizer(T5Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Adding SQL keywords to the tokenizer
        self.sql_keywords = [
            "SELECT", "FROM", "WHERE", "COUNT", "AND", "OR", "GROUP BY", "ORDER BY", 
            "INSERT", "UPDATE", "DELETE", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", 
            "FULL JOIN", "HAVING", "DISTINCT", "LIKE", "IN", "BETWEEN", "AS", 
            "CREATE", "ALTER", "DROP", "TABLE", "VIEW", "INDEX", "DATABASE", 
            "SET", "VALUES", "RETURNING", "UNION", "EXCEPT", "CASE", "WHEN", 
            "THEN", "ELSE", "END", "NULL", "NOT", "IS", "EXISTS", "LIMIT", 
            "OFFSET", "FETCH", "WITH", "ROLLBACK", "COMMIT", "TRANSACTION", 
            "GRANT", "REVOKE", "USE", "CHECK", "DEFAULT", "PRIMARY KEY", 
            "FOREIGN KEY", "UNIQUE", "INSERT INTO", "SELECT DISTINCT"
        ]
        self.add_tokens(self.sql_keywords)

    def _tokenize(self, text):
        tokens = super()._tokenize(text)  # Get initial tokens from the base class
        processed_tokens = []
        
        # Process the actual query to convert keywords to uppercase and tokenize
        actual_tokens = self._process_query(text)

        for token in tokens:
            # Strip leading underscore and process tokens
            cleaned_token = token.lstrip('▁')
            # Check for SQL keywords regardless of case
            if cleaned_token.lower() in (keyword.lower() for keyword in self.sql_keywords):
                processed_tokens.append(cleaned_token.upper())  # Convert to uppercase
            elif cleaned_token.isnumeric():
                processed_tokens.append("<value>")  # Placeholder for tokenization
            elif cleaned_token in self.special_tokens_map.values():  # Check if it's a special token
                processed_tokens.append(cleaned_token)
            else:
                processed_tokens.append(cleaned_token)

        return processed_tokens, actual_tokens  # Return both processed tokens and the tokenized actual query

    def _process_query(self, query):
        # Tokenize and convert SQL keywords to uppercase in the actual query
        words = query.split()
        processed_tokens = []
        for word in words:
            # Check if the word is a SQL keyword
            if word.upper() in self.sql_keywords:
                processed_tokens.append(word.upper())  # Convert to uppercase
            else:
                processed_tokens.append(word)  # Keep actual values unchanged
        return processed_tokens  # Return the list of processed tokens for the actual query

# Example of using the custom tokenizer
tokenizer = CustomT5Tokenizer.from_pretrained("t5-base")

sample_query = "select name, age FROM employees WHERE age > 30"
tokens, actual_tokens = tokenizer._tokenize(sample_query)

print("Tokenized with placeholders:", tokens)  # Output the tokens with placeholders
print("Actual query tokenized:", actual_tokens)  # Output the actual SQL query with keywords in caps and actual values
