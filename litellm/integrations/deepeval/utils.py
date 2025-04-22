def _prepare_input_str(input):
        """
        Stringify input. 
        """
        input_str = ""
        for message in input:
            input_str += f"{message['role']}: {message['content']}\n"
        return input_str