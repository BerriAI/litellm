from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, NumberAttribute, MapAttribute
from pynamodb.connection import Connection
from pynamodb.exceptions import DoesNotExist


class DynamoDBWrapper:
    def __init__(self, host="", region="us-east-2", **config):
        """
        Initialize the DynamoDB connection.
        """
        self.connection = Connection(region=region, host=host)
    
    def create_table(self, model_class):
        """
        Create a DynamoDB table based on the provided model class
        """
        if not model_class.exists():
            model_class.create_table(wait=True)
    
    def create_row(self, model_class, **attributes):
        """
        Create a row in a DynamoDB table based on the provided model class and attributes
        """
        item = model_class(**attributes)
        item.save()
        return item
    
    def read_row(self, model_class, hash_key, range_key=None):
        """
        Read a row from the DynamoDB table given a hash key (and optionally a range key)
        """
        try:
            if range_key is not None:
                item = model_class.get(hash_key, range_key=range_key)
            else:
                item = model_class.get(hash_key)
            return item
        except DoesNotExist:
            return None
    
    def update_row(self, model_class, hash_key, range_key=None, **updates):
        """
        Update a row in the table given a hash key (and optionally a range key), and updates mapping
        """
        item = self.read_row(model_class, hash_key, range_key)
        if item:
            for attribute, value in updates.items():
                if hasattr(item, attribute):
                    setattr(item, attribute, value)
            item.save()
            return item
        return None


# Define a user model using PynamoDB for demonstration purposes
class UserModel(Model):
    class Meta:
        table_name = "Users"
        region = 'us-west-2'
        billing_mode = 'PAY_PER_REQUEST'
    user_id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    age = NumberAttribute()

# Usage example
db_wrapper = DynamoDBWrapper()

# Create the table
db_wrapper.create_table(UserModel)

# Create a user
user = db_wrapper.create_row(UserModel, user_id="001", name="John Doe", age=30)

# Read the user data
retrieved_user = db_wrapper.read_row(UserModel, "001")

# Update the user data
updated_user = db_wrapper.update_row(UserModel, "001", name="Jane Doe", age=32)

print(retrieved_user.name, retrieved_user.age)  # Output: John Doe 30
print(updated_user.name, updated_user.age)      # Output: Jane Doe 32