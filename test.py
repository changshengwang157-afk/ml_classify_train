from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./outputs/model",
    tokenizer="./outputs/model"
)

# Try some text
examples = [
    "It was interesting at first, but the results weren't clear, and overall it wasn't very fun.",
    "It wasn't fun at first, but the ending was fun. The overall plot wasn't fun, but the love story was interesting. Overall, it's a positive story.",

]

predictions = classifier(examples)
print(predictions)


