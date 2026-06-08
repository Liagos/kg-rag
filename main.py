from kg_rag.rag.qa import ask


def run():

    print("\nRAG ready. Try queries:")

    while True:

        q = input("\nAsk: ").strip()

        if q.lower() in {"exit", "quit"}:
            break

        answer, meta = ask(q, mode="hybrid")

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    run()
