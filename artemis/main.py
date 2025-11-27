from src.agent.assistant import create_assistant


def main():
    print("Artemis simulation!")
    messages = []
    bot = create_assistant()
    while True:
        query = input('\nUser: ')
        if query.lower() in ['exit', 'quit']:
            break

        messages.append({'role': 'user', 'content': query})

        response = []
        for response_chunk in bot.run(messages=messages):
            response.append(response_chunk)
            print(response_chunk, end='', flush=True)

        messages.extend(response)
        print()  # New line after response


if __name__ == "__main__":
    main()
