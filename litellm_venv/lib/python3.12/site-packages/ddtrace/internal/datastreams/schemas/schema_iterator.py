import abc


class SchemaIterator:
    @abc.abstractmethod
    def iterate_over_schema(self, builder):
        pass
