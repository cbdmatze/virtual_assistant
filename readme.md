

'''

# 🚀 BULLS ⚛️ AI Chatbot Application - Readme for Grading 🚀

Welcome, esteemed teachers, to the BULLS ⚛️ AI Chatbot application! This project is designed to showcase a versatile and powerful chatbot platform that leverages the strengths of multiple cutting-edge Large Language Models (LLMs).  This readme will guide you through the setup, usage, and key features of the application, making it easy for you to evaluate its functionalities and innovations.

## ✨ Key Features & Grading Highlights ✨

Before diving into the technical setup, let's highlight the core features that make this application stand out and are crucial for your grading consideration:

*   **🧠 Multi-Model Mastery (60+ Models!):** While **ChatGPT is undeniably brilliant**, this application goes beyond by integrating a diverse range of over **60+ LLMs** from OpenAI, Anthropic (Claude), Groq, and Google (Gemini)!  This unique feature allows users to tap into a multitude of AI perspectives within a single platform. Imagine the productivity boost from getting insights from so many varied AI minds!  This is a core differentiator and a significant value proposition.

*   **🗣️ Conversation History:**  Every interaction is thoughtfully stored in a MySQL database, allowing users to revisit and review their previous chats. This promotes continuity and makes the application a useful tool for ongoing tasks and reflection.

*   **🌡️ Temperature Control for Creative Customization:**  Users can adjust the "temperature" parameter for text-based models. This isn't just a technical detail; it's a powerful way to control the AI's creativity and focus. Lower temperatures yield more focused and deterministic responses, ideal for factual queries. Higher temperatures encourage more creative and exploratory outputs, perfect for brainstorming or creative writing. This feature allows for nuanced interaction and output customization.

*   **🖼️ Image Generation:**  Beyond text, the application includes an image generation feature powered by a RapidAPI integration. Users can bring their imaginations to life with AI-generated visuals directly within the chat interface.

*   **👁️ Image-to-Text (OCR):**  Users can upload images, and the application uses Optical Character Recognition (OCR) via Tesseract to extract text from them. This text is then stored as a conversation, demonstrating practical utility beyond just chatbot interactions.

*   **🎨 User-Friendly Interface with Theme & Brightness Control:**  The React frontend provides a clean and intuitive user experience.  Users can switch between light and dark themes and even adjust the brightness of the application to suit their viewing preferences and environment.

## 🛠️ Setup Instructions - Get the Application Running! 🛠️

Follow these step-by-step instructions to set up and run the BULLS ⚛️ AI Chatbot application on your local machine.

### ⚙️ Prerequisites ⚙️

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.8+:**  Required for the backend (FastAPI). You can download it from [python.org](https://www.python.org/).
2.  **Node.js and npm:** Required for the frontend (React). Download from [nodejs.org](https://nodejs.org/).
3.  **MySQL Database:** You'll need a MySQL server running. You can download MySQL Community Server from [mysql.com](https://www.mysql.com/).
4.  **Tesseract OCR Engine:** For the image-to-text feature. Install Tesseract and its language data (English recommended). Instructions can be found here: [tesseract-ocr.github.io](https://tesseract-ocr.github.io/).  Make sure `tesseract` is added to your system's PATH environment variable.

### 📦 Backend Setup (FastAPI) 📦

1.  **Clone the Repository:**
    Open your terminal and clone the project repository to your local machine:

    ```bash
    git clone <repository_url>  # Replace <repository_url> with the actual repository URL
    cd <repository_folder>/backend
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's best practice to create a virtual environment to isolate project dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Backend Dependencies:**
    Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Database Setup (MySQL):**
    *   **Create a Database:** Log in to your MySQL server (e.g., using MySQL Workbench or command-line client) and create a database named `tasos` (as defined in the backend code):

        ```sql
        CREATE DATABASE tasos;
        ```

    *   **User Credentials:**  The backend code is configured to connect to MySQL with the following credentials:
        *   **Host:** `localhost`
        *   **User:** `matthias`
        *   **Password:** `Ma-294022275`
        *   **Database:** `tasos`

        **Important:** You need to either:
        *   **Change these credentials** in the `backend/app.py` file to match your MySQL setup. Look for the `mysql.connector.connect` section and update the `user`, `password`, and `database` parameters.
        *   **Create a MySQL user** named `matthias` with the password `Ma-294022275` and grant it access to the `tasos` database on your local MySQL server.

5.  **API Keys:**
    *   The backend code includes placeholder API keys for OpenAI, Anthropic, Groq, and Google Generative AI. **You MUST replace these with your own API keys** to use these services.
    *   **Obtain API Keys:**
        *   **OpenAI:**  [platform.openai.com](https://platform.openai.com/)
        *   **Anthropic:** [console.anthropic.com](https://console.anthropic.com/)
        *   **Groq:** [console.groq.com](https://console.groq.com/)
        *   **Google Generative AI:** [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
        *   **RapidAPI Key (for Image Generation):** You'll need a RapidAPI account and subscribe to the "ChatGPT Vision" API (or a similar text-to-image API) on [RapidAPI](https://rapidapi.com/). Replace the placeholder RapidAPI key in `backend/app.py`.

    *   **Update `backend/app.py`:** Open `backend/app.py` and find these lines. Replace the placeholder strings with your actual API keys:

        ```python
        openai.api_key = "YOUR_OPENAI_API_KEY"
        anthropic_api_key = "YOUR_ANTHROPIC_API_KEY"
        groq_api_key = "YOUR_GROQ_API_KEY"
        google_api_key = "YOUR_GOOGLE_API_KEY"
        headers = {
            "x-rapidapi-key": "YOUR_RAPIDAPI_KEY", # For Image Generation
            "x-rapidapi-host": "chatgpt-vision1.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        ```

6.  **Run the Backend:**
    Start the FastAPI backend server:

    ```bash
    uvicorn app:app --reload
    ```

    You should see output indicating the server is running, typically at `http://127.0.0.1:8000`.

### ⚛️ Frontend Setup (React) ⚛️

1.  **Navigate to Frontend Directory:**
    Open a **new terminal window** (leave the backend running in the first one) and navigate to the frontend directory:

    ```bash
    cd <repository_folder>/frontend
    ```

2.  **Install Frontend Dependencies:**
    Install the required npm packages:

    ```bash
    npm install
    ```

3.  **Run the Frontend:**
    Start the React development server:

    ```bash
    npm start
    ```

    This will usually open the application in your browser at `http://localhost:3000`. If it doesn't open automatically, navigate to this address in your web browser.

### 🎉 Access the Application! 🎉

With both the backend and frontend running, you should now be able to access the BULLS ⚛️ AI Chatbot application in your browser at `http://localhost:3000`.

## 🚀 Usage Guide - Exploring the Application 🚀

Here's a quick guide on how to use the application:

1.  **Login/Register:**
    *   On the homepage, you'll find "Login" and "Register" forms.
    *   If you're a new user, register a username and password.
    *   Existing users can log in with their credentials.

2.  **Text Chat:**
    *   After logging in, navigate to the "Chat" section.
    *   **API Provider Selection:** Choose between "OpenAI," "Anthropic," "Groq," or "Google" to select the LLM provider you want to use.
    *   **Model Selection:** Select a specific model from the dropdown menu. Notice the wide variety of models available!
    *   **Temperature Adjustment:** For adjustable temperature models, use the slider to set the temperature. Experiment to see how it affects the responses!
    *   **Prompt Input:** Enter your question or prompt in the text area.
    *   **Chat Button:** Click the "Chat" button to send your prompt and receive a response from the selected AI model.
    *   **Response Display:** The AI's response will be displayed in the "Response" section, beautifully formatted with Markdown and syntax highlighting for code.
    *   **Copy Response:** Use the "Copy Response" button to easily copy the AI's output.

3.  **Image Generation:**
    *   Switch to "Image Generation" mode using the mode toggle buttons above the chat area.
    *   **Prompt Input:** Describe the image you want to generate in the text area.
    *   **Dimension Control:** Adjust the width and height of the generated image if desired.
    *   **Generate Image Button:** Click "Generate Image" to create an image based on your prompt.
    *   **Image Display:** The generated image will be shown below the "Response" section.

4.  **Conversation History:**
    *   The "Conversations" section on the right side of the page displays your past conversations, ordered by most recent first.
    *   **View Conversations:** Click "Show Conversations" to expand and view your conversation history.
    *   **Delete Conversations:** You can delete individual conversations using the "Delete" button next to each conversation or delete all conversations using the "Delete All Conversations" button.
    *   **Timestamp & Model Info:** Each conversation entry shows the timestamp, model used, temperature (if applicable), and API provider, offering valuable context.

5.  **Image Upload (OCR):**
    *   Below the chat area, you'll find an "input type='file'" element.
    *   Click "Choose File" or drag and drop an image onto the chat area.
    *   The application will extract text from the image using OCR and display it as the response, also saving it to your conversation history.

6.  **Theme and Brightness Control:**
    *   In the header of the application, you can find controls to:
        *   Toggle between "Light Mode" and "Dark Mode".
        *   Adjust the "Brightness" of the application interface using a slider.

## 💻 Technology Stack 💻

This application utilizes the following technologies:

**Backend (FastAPI):**

*   **Framework:** FastAPI (Python - for building APIs quickly and efficiently)
*   **Language:** Python
*   **Database:** MySQL (for persistent storage of user data and conversations)
*   **LLM Integrations:**
    *   OpenAI Python Library
    *   Anthropic Python SDK
    *   Groq Python SDK
    *   Google Generative AI Python Library
*   **OCR:** Pytesseract (Python wrapper for Tesseract OCR)
*   **Security:** Werkzeug (for password hashing), FastAPI Security (OAuth2 Password Bearer)
*   **HTTP Requests:** Requests library (for image generation API)
*   **Asynchronous Operations:**  `async` and `await` throughout the backend for efficient handling of API calls.
*   **Web Server:** Uvicorn (ASGI server for running FastAPI applications)
*   **CORS:** FastAPI's `CORSMiddleware` for handling Cross-Origin Resource Sharing to allow frontend access.

**Frontend (React):**

*   **Framework:** React (for building a dynamic and interactive user interface)
*   **Libraries:**
    *   Axios (for making HTTP requests to the backend API)
    *   React Markdown (for rendering Markdown formatted responses)
    *   React Syntax Highlighter (for code syntax highlighting in responses)
    *   React Loader Spinner (for visual feedback during API loading)
*   **Styling:** CSS (for application styles and layout)

## 🎓 Grading Considerations for Teachers 🎓

When evaluating this application, please consider the following aspects:

*   **Functionality (Core Features Working):**
    *   Successful user registration and login.
    *   Text-based chat functionality with all integrated LLM providers (OpenAI, Anthropic, Groq, Google).
    *   Accurate model selection and temperature parameter application.
    *   Correct display of AI responses with Markdown and syntax highlighting.
    *   Working image generation feature.
    *   Functional conversation history storage, retrieval, and deletion.
    *   Image upload and OCR text extraction.
    *   Theme and brightness control responsiveness.

*   **Code Quality and Structure:**
    *   Well-organized backend code using FastAPI principles.
    *   Clear separation of concerns in both backend and frontend.
    *   Use of logging for debugging and error tracking.
    *   Input validation and error handling in the backend.
    *   Secure password hashing for user registration.

*   **Feature Implementation and Innovation:**
    *   **Multi-Model Integration:**  The core strength of integrating multiple LLM APIs seamlessly.
    *   **Temperature Control:**  Effective implementation of temperature adjustment for text models.
    *   **Conversation Persistence:**  Database storage of conversations for user benefit.
    *   **Image Generation and OCR:**  Bonus features adding practical utility.

*   **User Interface (UI) and User Experience (UX):**
    *   Intuitive and user-friendly frontend design.
    *   Responsive layout and smooth interactions.
    *   Clear presentation of AI responses and conversation history.
    *   Theme and brightness customization options.

*   **Documentation (Readme.md):**
    *   Comprehensive and clear setup instructions.
    *   Detailed explanation of features and usage.
    *   Clear technology stack overview.

## 🎉 Conclusion 🎉

Thank you for taking the time to evaluate the BULLS ⚛️ AI Chatbot application. We believe this project effectively demonstrates the power and versatility of combining multiple AI models into a single, user-friendly platform.  The ability to access over 60+ LLM opinions, customize AI creativity with temperature control, and maintain a history of interactions makes this application a valuable tool for productivity and exploration. We hope you find it insightful and appreciate the effort put into its development.

Happy Grading! 🚀

'''
