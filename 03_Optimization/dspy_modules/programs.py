import dspy
from .signatures import TransactionExtractionSignature

class TransactionExtractor(dspy.Module):
    def __init__(self, use_cot: bool = False):
        """
        Initialize the TransactionExtractor module.
        
        Args:
            use_cot (bool): If True, uses ChainOfThought. Otherwise uses Predict.
        """
        super().__init__()
        if use_cot:
            self.extractor = dspy.ChainOfThought(TransactionExtractionSignature)
        else:
            self.extractor = dspy.Predict(TransactionExtractionSignature)

    def forward(self, message: str):
        result = self.extractor(message=message)
        return dspy.Prediction(financial_transactions_json=result.financial_transactions_json)
