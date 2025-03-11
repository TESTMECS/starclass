from transformers import pipeline
import sys


labels = {
    "LABEL_0": "Action",
    "LABEL_1": "Result",
    "LABEL_2": "Situation",
    "LABEL_3": "Task",
}
if __name__ == "__main__":
    # Check for command-line argument
    if len(sys.argv) > 1:
        sample = sys.argv[1]
    else:
        sample = "I've trained the model and used the data"
    classifier = pipeline(
        "text-classification", model="dnttestmee/starclass_modernbert"
    )

    output = classifier(sample)
    print(f"Input: {sample}")
    print(f"Output: {labels[str(output[0]['label'])]}")
    print(f"Score: {output[0]['score']}")
# FOR MULTIPLE SENTENCES
# classifer = pipeline("text-classification", model="dnttestmee/starclass_bert")
# classifer(["This restaurant is awesome", "This restaurant is awful"])
