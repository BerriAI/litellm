from litellm.integrations.custom_logger import CustomLogger


class CloudZeroLogger(CustomLogger):
    async def export_usage_data(self):
        """
        Exports the usage data to CloudZero.

        - Reads data from the DB
        - Transforms the data to the CloudZero format
        - Sends the data to CloudZero
        """
        pass
    

    async def dry_run_export_usage_data(self):
        """
        Only prints the data that would be exported to CloudZero.
        """
        pass
