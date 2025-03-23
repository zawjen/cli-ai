from sdk.args import Args
from sdk.query_processor import QueryProcessor
from transformers import pipeline


if __name__ == "__main__":
    args = Args()
    
    processor = QueryProcessor(args.query)

    answer = processor.generate_answer()

    processor.save_answer()

    print(f"Answer: {answer.text}")


