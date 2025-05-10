import speech_recognition as sr

keywords = ["yes", "no", "up", "down", "stop", "go", "left", "right", "start", "exit"]

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak a word...")
    audio = recognizer.listen(source)

    try:
        result = recognizer.recognize_google(audio).lower()
        print("You said:", result)

        if result in keywords:
            print("Recognized keyword:", result)
        else:
            print("Word not in predefined list.")

    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print("Error from Google Speech Recognition service; {0}".format(e))
