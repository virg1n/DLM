from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from transformers import AutoTokenizer
import secrets

def datatrove_tokenization_executor(hf_dataset_id,
                                    name,
                                    id_column,
                                    text_column,
                                    output_folder,
                                    tokenizer_id,
                                    eos_token,
                                    num_workers,
                                    max_documents=None, # max_documents parameter
                                    job_id=None):
    if not job_id:
        job_id = secrets.token_hex(8)

    pipeline = [
        HuggingFaceDatasetReader(
            dataset=hf_dataset_id,
            dataset_options={
                "split": 'train',
                "name": name,
                "columns": [id_column, text_column]
            },
            text_key=text_column,
            id_key=id_column,
            streaming=True,
            limit=max_documents
        ),
        DocumentTokenizer(
            output_folder=output_folder,
            tokenizer_name_or_path=tokenizer_id,
            eos_token=eos_token,
            batch_size=1000,
            max_tokens_per_file=int(1e8),
            seed=1998
        )
    ]
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"logs_{job_id}/",
        tasks=num_workers,
        workers=num_workers,
    )

    return executor


def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    doc_limit = 30

    executor = datatrove_tokenization_executor(
        job_id="tokenize_fineweb-edu_sample10BT",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        id_column="id",
        text_column="text",
        output_folder="./fwe-10BT",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        num_workers=2,
        max_documents=doc_limit 
    )
    executor.run()

if __name__ == "__main__":
    main()