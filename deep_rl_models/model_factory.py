class ModelFactory:
    @classmethod
    def get_model(cls, config):
        raise NotImplementedError("Subclasses of ModelFactory must implement get_model() to return the mode")
