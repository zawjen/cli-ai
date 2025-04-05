import time
from sdk.args import Args
from sdk.query_processor import QueryProcessor
from transformers import pipeline

from sdk.time_logger import TimeLogger


if __name__ == "__main__":
    start_time = time.time()

    args = Args()
    
    processor = QueryProcessor(args.query)

    answer = processor.generate_answer()

    processor.save_answer()

    TimeLogger.log("FINISHED: Question answered", start_time)

    print(f"[Question]: {args.query.text}")
    print(f"[Answer]: {answer.text}")




