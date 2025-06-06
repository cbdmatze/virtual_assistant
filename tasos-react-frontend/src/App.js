// npm install react react-dom axios react-markdown react-syntax-highlighter react-loader-spinner
// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialOceanic } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Oval } from 'react-loader-spinner';
import './App.css';

function App() {
    const [loginUsername, setLoginUsername] = useState('');
    const [loginPassword, setLoginPassword] = useState('');
    const [registerUsername, setRegisterUsername] = useState('');
    const [registerPassword, setRegisterPassword] = useState('');
    const [token, setToken] = useState('');
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');
    const [model, setModel] = useState('gpt-3.5-turbo');
    const [temperature, setTemperature] = useState(0.7);
    const [conversations, setConversations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [loginMessage, setLoginMessage] = useState('');
    const [registerMessage, setRegisterMessage] = useState('');
    const [showConversations, setShowConversations] = useState(true);
    const [image, setImage] = useState(null);
    const [theme, setTheme] = useState('light'); // Default theme
    const [apiProvider, setApiProvider] = useState('openai'); // 'openai', 'anthropic', 'groq', 'google', 'huggingface', 'langchain', or 'langgraph'
    const [brightness, setBrightness] = useState(1.0); // Default brightness level (0 = darkest, 1 = lightest)
  
    // Image generation states
    const [generationMode, setGenerationMode] = useState('text'); // 'text' or 'image'
    const [generatedImage, setGeneratedImage] = useState(null);
    const [imageWidth, setImageWidth] = useState(512);
    const [imageHeight, setImageHeight] = useState(512);
    
    // YouTube video states
    const [embeddedVideo, setEmbeddedVideo] = useState(null);
    const [videoSearchResults, setVideoSearchResults] = useState([]);
    const [showVideoSearch, setShowVideoSearch] = useState(false);
    const [videoPlayerWidth, setVideoPlayerWidth] = useState(640);
    const [videoPlayerHeight, setVideoPlayerHeight] = useState(360);

    // Define models with fixed temperature
    const fixedTemperatureModels = ['o1-mini-2024-09-12', 'o1-preview-2024-09-12', 'o1-mini', 'o1-preview'];
    
    // Define models with adjustable temperature
    const adjustableTemperatureModels = [
        'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o-2024-11-20',
        'gpt-4o-mini-2024-07-18', 'gpt-4-0125-preview', 'gpt-4o-2024-08-06', 'gpt-4o',
        'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4o--2024-05-13',
        'chatgpt-4o-latest', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k', 'gpt-4-1106-preview'
    ];
   
    // Define Anthropic models
    const anthropicModels = [
        'claude-3-5-sonnet-latest',
        'claude-3-7-sonnet-20250219',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'claude-3-5-sonnet-20240620',
        'claude-3-haiku-20240307'
    ];

    // Define Groq models
    const groqModels = [
        'llama-3.3-70b-versatile',
        'qwen-2.5-32b',
        'llama-3.2-1b-preview',
        'gemma2-9b-it',
        'mixtral-8x7b-32768',
        'deepseek-r1-distill-llama-70b',
        'qwen-2.5-coder-23b',
        'llama3-8b-8192',
        'llama-3.2-11b-vision-preview',
        'llama-3.2-90b-vision-preview'
    ];
    
    // Define Google Gemini models
    const googleModels = [
        'gemini-1.5-flash',
        'gemini-1-5-flash-002',
        'gemini-1.5-flash-8b',
        'gemini-1-5-flash-8b-001',
        'gemini-1.5-flash-8b-latest',
        'gemini-1.5-flash-8b-exp-0827',
        'gemini-1.5-flash-8b-exp-0927',
        'gemini-2.0-flash-exp',
        'gemini-2.0-flash',
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
        'gemini-1.5-pro',
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash-001',
        'gemini-1.5-flash-001-tuning',
        'gemini-2.0-flash-001',
        'gemini-2.0-flash-lite-001',
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash-lite-preview-02-05',
        'gemini-2.0-flash-lite-preview',
        'gemini-2.0-pro-exp',
        'gemini-2.0-pro-exp-02-05',
        'gemini-exp-1206',
        'gemini-2.0-flash-thinking-exp-01-21',
        'gemini-2.0-flash-thinking-exp-1219',
        'learnlm-1.5-pro-experimental'
    ];

    // Define HuggingFace models
    const huggingfaceModels = [
        'gpt2',
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl',
        'distilgpt2'
    ];

    // Define LangChain models
    const langchainModels = [
        'google-gemini',
        'gemini-1.5-pro',
        'gemini-1.5-flash'
    ];

    // Define LangGraph models
    const langgraphModels = [
        'google-gemini-graph',
        'gemini-1.5-pro',
        'gemini-1.5-flash'
    ];

    // Check if model has fixed temperature
    const isFixedTemperatureModel = fixedTemperatureModels.includes(model);
    
    // Determine current model provider
    const isAnthropicModel = anthropicModels.includes(model);
    const isGroqModel = groqModels.includes(model);
    const isGoogleModel = googleModels.includes(model);
    const isHuggingFaceModel = huggingfaceModels.includes(model);
    const isLangChainModel = langchainModels.includes(model);
    const isLangGraphModel = langgraphModels.includes(model);

    // Get temperature range based on API provider
    const getTemperatureRange = () => {
        if (apiProvider === 'anthropic') {
            return { min: -1, max: 1 }; // Anthropic range is -1 to 1
        } else {
            return { min: -1, max: 2 }; // Default range for OpenAI, Groq and Google
        }
    };

    const temperatureRange = getTemperatureRange();

    // Modified useEffect to coordinate brightness and theme
    useEffect(() => {
        document.documentElement.style.setProperty('--brightness', brightness);
        
        // Update theme based on brightness value
        if (brightness > 0.5 && theme !== 'light') {
            setTheme('light');
        } else if (brightness <= 0.5 && theme !== 'dark') {
            setTheme('dark');
        }
        
        document.body.className = theme;
    }, [brightness, theme]);

    // Modified theme toggle handler
    const handleThemeToggle = () => {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        
        // Set brightness to match the theme
        setBrightness(newTheme === 'light' ? 1.0 : 0.0);
    };

    // Reset temperature and set API provider based on the selected model
    useEffect(() => {
        if (isFixedTemperatureModel) {
            setTemperature(1);
        }
        
        if (isAnthropicModel) {
            setApiProvider('anthropic');
            // Ensure temperature is within Anthropic's valid range (-1 to 1)
            if (temperature > 1) {
                setTemperature(1);
            }
        } else if (isGroqModel) {
            setApiProvider('groq');
        } else if (isGoogleModel) {
            setApiProvider('google');
        } else if (isHuggingFaceModel) {
            setApiProvider('huggingface');
        } else if (isLangChainModel) {
            setApiProvider('langchain');
        } else if (isLangGraphModel) {
            setApiProvider('langgraph');
        } else {
            setApiProvider('openai');
        }
    }, [model, isFixedTemperatureModel, isAnthropicModel, isGroqModel, isGoogleModel, isHuggingFaceModel, isLangChainModel, isLangGraphModel, temperature]);
    
    // Clear generated image when switching modes
    useEffect(() => {
        setGeneratedImage(null);
    }, [generationMode]);

    const handleLogin = async () => {
        try {
            const formData = new URLSearchParams();
            formData.append('username', loginUsername);
            formData.append('password', loginPassword);

            const res = await axios.post('http://localhost:8000/token', formData, {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });
            setToken(res.data.access_token);
            setLoginMessage('Login successful!');
            fetchConversations();
        } catch (error) {
            console.error('Login failed', error);
            setLoginMessage('Login failed. Please check your username and password.');
        }
    };

    const handleRegister = async () => {
        try {
            await axios.post('http://localhost:8000/register', {
                username: registerUsername,
                password: registerPassword
            });
            setRegisterMessage('Registration successful! You can now log in.');
        } catch (error) {
            console.error('Registration failed', error);
            setRegisterMessage('Registration failed. Username may already exist.');
        }
    };

    const handleChat = async () => {
        setLoading(true);
        try {
            const currentTemperature = isFixedTemperatureModel ? 1 : temperature;
            console.log("Sending chat request:", {
                prompt,
                model,
                temperature: currentTemperature,
                api_provider: apiProvider
            });
            
            const res = await axios.post('http://localhost:8000/chat', {
                prompt,
                model,
                temperature: currentTemperature,
                api_provider: apiProvider
            }, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            // Handle the response
            let responseText = res.data.response;
            
            // Check if we got video_embed data directly from the API
            if (res.data.video_embed) {
                console.log("Received embedded video data:", res.data.video_embed);
                setEmbeddedVideo({
                    videoId: res.data.video_embed.video_id,
                    embedHtml: res.data.video_embed.embed_html,
                    width: videoPlayerWidth,
                    height: videoPlayerHeight
                });
                // Set response and continue
                setResponse(responseText);
            }
            // Also check for the older format with the video metadata in the response text
            else if (responseText.startsWith("VIDEO_RESPONSE_TYPE=")) {
                // Parse video response metadata
                const metaEndIndex = responseText.indexOf("|", responseText.indexOf("|") + 1) + 1;
                const metadataStr = responseText.substring(0, metaEndIndex);
                const cleanResponse = responseText.substring(metaEndIndex);
                
                // Extract video type and ID
                const videoType = metadataStr.match(/VIDEO_RESPONSE_TYPE=([^|]+)/)[1];
                const videoId = metadataStr.match(/VIDEO_ID=([^|]+)/)[1];
                
                // Handle different video types
                if (videoType === "embedded") {
                    // Fetch the embedded player
                    const embedRes = await axios.post(`http://localhost:8000/videos/${videoId}/embed`, {
                        width: videoPlayerWidth,
                        height: videoPlayerHeight
                    }, {
                        headers: { Authorization: `Bearer ${token}` }
                    });
                    
                    if (embedRes.data.success) {
                        setEmbeddedVideo({
                            videoId: videoId,
                            embedHtml: embedRes.data.embed_html,
                            width: videoPlayerWidth,
                            height: videoPlayerHeight
                        });
                    }
                } else if (videoType === "downloaded") {
                    // Handle downloaded video
                    const videoRes = await axios.post(`http://localhost:8000/videos/${videoId}/download`, {}, {
                        headers: { Authorization: `Bearer ${token}` }
                    });
                    
                    if (videoRes.data.success) {
                        // Add logic to handle downloaded video
                        console.log("Video downloaded:", videoRes.data);
                    }
                }
                
                // Set the cleaned response without the metadata
                setResponse(cleanResponse);
            } else {
                // Regular text response
                setResponse(responseText);
                setEmbeddedVideo(null);
            }
            
            fetchConversations();
        } catch (error) {
            console.error('Chat failed', error);
            const errorDetail = error.response?.data?.detail || error.message;
            console.log("Error detail:", errorDetail);
            setResponse(`Error: ${errorDetail}`);
            setEmbeddedVideo(null);
        } finally {
            setLoading(false);
        }
    };
    
    const handleYouTubeSearch = async (query) => {
        setLoading(true);
        try {
            const res = await axios.get(`http://localhost:8000/youtube/search?query=${encodeURIComponent(query)}`, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            if (res.data.success) {
                setVideoSearchResults(res.data.results);
                setShowVideoSearch(true);
            } else {
                setResponse(`Error searching YouTube: ${res.data.error}`);
            }
        } catch (error) {
            console.error('YouTube search failed', error);
            setResponse(`Error searching YouTube: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };
    
    const handleWatchVideo = async (videoId) => {
        try {
            const res = await axios.get(`http://localhost:8000/youtube/watch/${videoId}`, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            if (res.data.success) {
                setResponse(`Opening video ${videoId} in your browser.`);
            } else {
                setResponse(`Error opening video: ${res.data.error}`);
            }
        } catch (error) {
            console.error('Error opening video', error);
            setResponse(`Error opening video: ${error.message}`);
        }
    };
    
    // Updated to prevent fetching conversations after embedding
    const handleEmbedVideo = async (videoId) => {
        try {
            // First, clear any existing embedded video
            setEmbeddedVideo(null);
            
            const res = await axios.post(`http://localhost:8000/videos/${videoId}/embed`, {
                width: videoPlayerWidth,
                height: videoPlayerHeight
            }, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            if (res.data.success) {
                setEmbeddedVideo({
                    videoId: videoId,
                    embedHtml: res.data.embed_html,
                    width: videoPlayerWidth,
                    height: videoPlayerHeight
                });
                // Removed the fetchConversations call to prevent double video issue
                // The conversation will be fetched the next time user performs another action
            } else {
                setResponse(`Error embedding video: ${res.data.error}`);
            }
        } catch (error) {
            console.error('Error embedding video', error);
            setResponse(`Error embedding video: ${error.message}`);
        }
    };
    
    const handleImageGeneration = async () => {
        setLoading(true);
        setGeneratedImage(null);
        try {
            console.log("Sending image generation request:", {
                prompt,
                width: imageWidth,
                height: imageHeight,
                steps: 1
            });
            
            const res = await axios.post('http://localhost:8000/generate-image', {
                prompt,
                width: imageWidth,
                height: imageHeight,
                steps: 1
            }, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            console.log("Image generation response:", res.data);
            
            if (res.data.success) {
                setResponse(res.data.response);
                setGeneratedImage(res.data.image);
                fetchConversations();
            } else {
                setResponse("Failed to generate image: " + res.data.response);
            }
        } catch (error) {
            console.error('Image generation failed', error);
            const errorDetail = error.response?.data?.detail || error.message;
            console.log("Image generation error detail:", errorDetail);
            setResponse("Error generating image: " + (error.response?.data?.detail || error.message));
        } finally {
            setLoading(false);
        }
    };

    const fetchConversations = async () => {
        if (!token) return;
        
        try {
            const res = await axios.get('http://localhost:8000/conversations', {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            // Process any video conversations to make them playable
            const processedConversations = res.data.conversations.map(conv => {
                // Check if this is a YouTube-related conversation
                if (conv.api_provider === 'youtube' || conv.video_id) {
                    return {
                        ...conv,
                        has_video: true,
                        // Add more video-specific data as needed
                    };
                }
                return conv;
            });
            
            setConversations(processedConversations);
        } catch (error) {
            console.error('Fetching conversations failed', error);
        }
    };

    const deleteConversation = async (id) => {
        try {
            await axios.delete(`http://localhost:8000/conversations/${id}`, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            fetchConversations();
        } catch (error) {
            console.error('Deleting conversation failed', error);
        }
    };

    const deleteAllConversations = async () => {
        try {
            await axios.delete('http://localhost:8000/conversations', {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            fetchConversations();
        } catch (error) {
            console.error('Deleting all conversations failed', error);
        }
    };

    const handleImageUpload = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('http://localhost:8000/upload-image', formData, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResponse(res.data.ocr_text);
            fetchConversations();
        } catch (error) {
            console.error('Image upload failed', error);
        }
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        setImage(URL.createObjectURL(file));
        handleImageUpload({ target: { files: [file] } });
    };

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const formatTimestamp = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    };

    // Enhanced video player component to handle embedding safely
    const VideoPlayer = ({ embedHtml }) => {
        if (!embedHtml) return null;
        
        // Clean up the HTML to ensure it's just the iframe with proper attributes
        // This is a simple implementation - in production, you might want a more robust sanitizer
        const cleanHtml = embedHtml.trim();
        
        return (
            <div className="video-player-container">
                <div dangerouslySetInnerHTML={{ __html: cleanHtml }} />
            </div>
        );
    };

    // Updated to clear existing embedded video before playing a new one
    const playVideoFromHistory = async (videoDbId) => {
        try {
            // First, clear any existing embedded video
            setEmbeddedVideo(null);
            
            const res = await axios.get(`http://localhost:8000/videos/${videoDbId}`, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            
            if (res.data.success) {
                const videoData = res.data.video_data;
                
                if (videoData.type === "embedded") {
                    setEmbeddedVideo({
                        videoId: videoData.video_id,
                        embedHtml: videoData.embed_html,
                        width: videoData.width || videoPlayerWidth,
                        height: videoData.height || videoPlayerHeight
                    });
                } else if (videoData.type === "downloaded") {
                    // Handle playback of downloaded video
                    window.open(`http://localhost:8000/videos/${videoDbId}/stream`, '_blank');
                } else {
                    // Handle reference-only videos
                    window.open(`https://www.youtube.com/watch?v=${videoData.video_id}`, '_blank');
                }
            }
        } catch (error) {
            console.error('Error playing video from history', error);
            setResponse(`Error playing video: ${error.message}`);
        }
    };

    // Updated code component to ensure the copy button is visible with proper styling
    const components = {
        code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
                <div className="code-block">
                    <div className="code-header">
                        <button 
                            className="copy-code-button" 
                            onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                        >
                            Copy
                        </button>
                    </div>
                    <div className="code-content">
                        <SyntaxHighlighter 
                            style={materialOceanic} 
                            language={match[1]} 
                            PreTag="div" 
                            {...props}
                        >
                            {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                    </div>
                </div>
            ) : (
                <code className={className} {...props}>
                    {children}
                </code>
            );
        }
    };

    const copyResponse = () => {
        navigator.clipboard.writeText(response);
    };


    return (
        <div className="App">
            <div className="header">
                <h1>BULLS ⚛️ AI</h1>
                <div className="header-controls">
                    <div className="brightness-control">
                        <label>Brightness: {Math.round(brightness * 100)}%</label>
                        <input 
                            type="range" 
                            min="0"  
                            max="1"  
                            step="0.05"
                            value={brightness}
                            onChange={(e) => setBrightness(parseFloat(e.target.value))}
                        />
                    </div>
                    <button className="theme-toggle" onClick={handleThemeToggle}>
                        {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
                    </button>
                </div>
            </div>
            <div className="form-container">
                <div className="form">
                    <h2>Login</h2>
                    <div className="input-group">
                        <input type="text" placeholder="Username" value={loginUsername} onChange={(e) => setLoginUsername(e.target.value)} />
                        <input type="password" placeholder="Password" value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)} />
                        <button className="full-width-button" onClick={handleLogin}>Login</button>
                    </div>
                    {loginMessage && <p>{loginMessage}</p>}
                </div>
                <div className="form">
                    <h2>Register</h2>
                    <div className="input-group">
                        <input type="text" placeholder="Username" value={registerUsername} onChange={(e) => setRegisterUsername(e.target.value)} />
                        <input type="password" placeholder="Password" value={registerPassword} onChange={(e) => setRegisterPassword(e.target.value)} />
                        <button className="full-width-button" onClick={handleRegister}>Register</button>
                    </div>
                    {registerMessage && <p>{registerMessage}</p>}
                </div>
            </div>
            <div className="chat-container" onDrop={handleDrop} onDragOver={handleDragOver}>
                <h2>Chat</h2>
                
                {/* Main Mode Toggle */}
                <div className="mode-toggle">
                    <button 
                        className={`mode-button ${generationMode === 'text' ? 'active' : ''}`}
                        onClick={() => setGenerationMode('text')}
                    >
                        Text Chat
                    </button>
                    <button 
                        className={`mode-button ${generationMode === 'image' ? 'active' : ''}`}
                        onClick={() => setGenerationMode('image')}
                    >
                        Image Generation
                    </button>
                </div>
                
                {generationMode === 'text' ? (
                    /* Text chat UI */
                    <>
                        <div className="model-settings">
                            <div className="api-provider-selector">
                                <label>API Provider:</label>
                                <div className="api-toggle">
                                    <button 
                                        className={`api-button ${apiProvider === 'openai' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('openai');
                                            if (!adjustableTemperatureModels.includes(model) && !fixedTemperatureModels.includes(model)) {
                                                setModel('gpt-3.5-turbo');
                                            }
                                        }}
                                    >
                                        OpenAI
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'anthropic' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('anthropic');
                                            if (!anthropicModels.includes(model)) {
                                                setModel('claude-3-5-sonnet-latest');
                                            }
                                            // Ensure temperature doesn't exceed Anthropic's max
                                            if (temperature > 1) {
                                                setTemperature(1);
                                            }
                                        }}
                                    >
                                        Anthropic
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'groq' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('groq');
                                            if (!groqModels.includes(model)) {
                                                setModel('llama-3.2-90b-vision-preview');
                                            }
                                        }}
                                    >
                                        Groq
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'google' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('google');
                                            if (!googleModels.includes(model)) {
                                                setModel('gemini-1.5-pro');
                                            }
                                        }}
                                    >
                                        Google
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'huggingface' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('huggingface');
                                            setModel('gpt2');
                                        }}
                                    >
                                        HuggingFace
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'langchain' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('langchain');
                                            setModel('google-gemini');
                                        }}
                                    >
                                        LangChain
                                    </button>
                                    <button 
                                        className={`api-button ${apiProvider === 'langgraph' ? 'active' : ''}`}
                                        onClick={() => {
                                            setApiProvider('langgraph');
                                            setModel('google-gemini-graph');
                                        }}
                                    >
                                        LangGraph
                                    </button>
                                </div>
                            </div>
                           
                            {(apiProvider !== 'huggingface' && apiProvider !== 'langchain' && apiProvider !== 'langgraph') && (
                                <>
                                    <div className="model-selector">
                                        <label htmlFor="model-select">Model:</label>
                                        <select 
                                            id="model-select"
                                            value={model} 
                                            onChange={(e) => setModel(e.target.value)}
                                        >
                                            {apiProvider === 'openai' && (
                                                <>
                                                    <optgroup label="Fixed Temperature Models (1.0)">
                                                        {fixedTemperatureModels.map(model => (
                                                            <option key={model} value={model}>{model}</option>
                                                        ))}
                                                    </optgroup>
                                                    <optgroup label="Adjustable Temperature Models">
                                                        {adjustableTemperatureModels.map(model => (
                                                            <option key={model} value={model}>{model}</option>
                                                        ))}
                                                    </optgroup>
                                                </>
                                            )}
                                            
                                            {apiProvider === 'anthropic' && (
                                                <optgroup label="Claude Models">
                                                    {anthropicModels.map(model => (
                                                        <option key={model} value={model}>{model}</option>
                                                    ))}
                                                </optgroup>
                                            )}
                                            
                                            {apiProvider === 'groq' && (
                                                <optgroup label="Groq Models">
                                                    {groqModels.map(model => (
                                                        <option key={model} value={model}>{model}</option>
                                                    ))}
                                                </optgroup>
                                            )}
                                            
                                            {apiProvider === 'google' && (
                                                <optgroup label="Google Gemini Models">
                                                    {googleModels.map(model => (
                                                        <option key={model} value={model}>{model}</option>
                                                    ))}
                                                </optgroup>
                                            )}
                                        </select>
                                    </div>
                                    <div className="temperature-control">
                                        <label htmlFor="temperature-slider">
                                            Temperature: {temperature} {isFixedTemperatureModel && "(Fixed)"}
                                            {apiProvider === 'anthropic' && " (Range: -1 to 1)"}
                                            {apiProvider !== 'anthropic' && " (Range: -1 to 2)"}
                                        </label>
                                        <input 
                                            id="temperature-slider"
                                            type="range" 
                                            min={temperatureRange.min}
                                            max={temperatureRange.max}
                                            step="0.1"
                                            value={temperature}
                                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                                            disabled={isFixedTemperatureModel}
                                            className={isFixedTemperatureModel ? "disabled" : ""}
                                        />
                                    </div>
                                </>
                            )}
                            
                            {apiProvider === 'huggingface' && (
                                <div className="model-info-banner">
                                    <i className="fas fa-info-circle"></i>
                                    <span>Using local HuggingFace GPT-2 model</span>
                                </div>
                            )}
                            
                            {apiProvider === 'langchain' && (
                                <>
                                    <div className="model-selector">
                                        <label htmlFor="langchain-model-select">LangChain Model:</label>
                                        <select 
                                            id="langchain-model-select"
                                            value={model} 
                                            onChange={(e) => setModel(e.target.value)}
                                        >
                                            {langchainModels.map(model => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="model-info-banner">
                                        <i className="fas fa-search"></i>
                                        <span>Using LangChain with Google Gemini for enhanced search</span>
                                    </div>
                                </>
                            )}
                            
                            {apiProvider === 'langgraph' && (
                                <>
                                    <div className="model-selector">
                                        <label htmlFor="langgraph-model-select">LangGraph Model:</label>
                                        <select 
                                            id="langgraph-model-select"
                                            value={model} 
                                            onChange={(e) => setModel(e.target.value)}
                                        >
                                            {langgraphModels.map(model => (
                                                <option key={model} value={model}>{model}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="model-info-banner">
                                        <i className="fas fa-brain"></i>
                                        <span>Using LangGraph with Google Gemini for enhanced search with chain-of-thought reasoning</span>
                                    </div>
                                    
                                    {/* New - YouTube search controls */}
                                    <div className="youtube-controls">
                                        <div className="youtube-search">
                                            <button 
                                                className="youtube-search-button"
                                                onClick={() => handleYouTubeSearch(prompt)}
                                            >
                                                <i className="fab fa-youtube"></i> Search YouTube
                                            </button>
                                            <div className="video-player-size">
                                                <label>Player Size:</label>
                                                <select 
                                                    onChange={(e) => {
                                                        const [width, height] = e.target.value.split('x');
                                                        setVideoPlayerWidth(parseInt(width));
                                                        setVideoPlayerHeight(parseInt(height));
                                                    }}
                                                    value={`${videoPlayerWidth}x${videoPlayerHeight}`}
                                                >
                                                    <option value="426x240">426x240 (240p)</option>
                                                    <option value="640x360">640x360 (360p)</option>
                                                    <option value="854x480">854x480 (480p)</option>
                                                    <option value="1280x720">1280x720 (720p)</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                        
                        {/* YouTube search results section */}
                        {showVideoSearch && videoSearchResults.length > 0 && (
                            <div className="youtube-search-results">
                                <h3>YouTube Search Results</h3>
                                <button onClick={() => setShowVideoSearch(false)} className="close-results">Close Results</button>
                                <div className="video-results-grid">
                                    {videoSearchResults.map((video, index) => (
                                        <div key={index} className="video-result-card">
                                            <div className="video-thumbnail">
                                                <img 
                                                    src={video.thumbnail} 
                                                    alt={video.title} 
                                                    onClick={() => handleEmbedVideo(video.videoId)}
                                                />
                                            </div>
                                            <div className="video-info">
                                                <h4>{video.title}</h4>
                                                <p>{video.channel}</p>
                                                <div className="video-actions">
                                                    <button onClick={() => handleEmbedVideo(video.videoId)}>
                                                        Watch in App
                                                    </button>
                                                    <button onClick={() => handleWatchVideo(video.videoId)}>
                                                        Open in Browser
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        <textarea 
                            rows="10" 
                            placeholder={`Enter your prompt for ${
                                apiProvider === 'openai' ? 'OpenAI' : 
                                apiProvider === 'anthropic' ? 'Claude' : 
                                apiProvider === 'google' ? 'Gemini' : 
                                apiProvider === 'huggingface' ? 'HuggingFace GPT-2' :
                                apiProvider === 'langchain' ? 'LangChain + Gemini' :
                                apiProvider === 'langgraph' ? 'LangGraph + Gemini' : 'Groq'
                            }...`} 
                            value={prompt} 
                            onChange={(e) => setPrompt(e.target.value)} 
                        />
                        <button onClick={handleChat}>Chat with {
                            apiProvider === 'openai' ? 'OpenAI' : 
                            apiProvider === 'anthropic' ? 'Claude' : 
                            apiProvider === 'google' ? 'Gemini' : 
                            apiProvider === 'huggingface' ? 'HuggingFace GPT-2' :
                            apiProvider === 'langchain' ? 'LangChain + Gemini' :
                            apiProvider === 'langgraph' ? 'LangGraph + Gemini' : 'Groq'
                        }</button>
                    </>
                ) : (
                    /* Image generation UI */
                    <>
                        <div className="image-settings">
                            <div className="dimension-control">
                                <label>Width:</label>
                                <input 
                                    type="number" 
                                    min="256" 
                                    max="1024" 
                                    step="64" 
                                    value={imageWidth} 
                                    onChange={(e) => setImageWidth(parseInt(e.target.value))} 
                                />
                                <label>Height:</label>
                                <input 
                                    type="number" 
                                    min="256" 
                                    max="1024" 
                                    step="64" 
                                    value={imageHeight} 
                                    onChange={(e) => setImageHeight(parseInt(e.target.value))} 
                                />
                            </div>
                        </div>
                        <textarea 
                            rows="10" 
                            placeholder="Describe the image you want to generate..." 
                            value={prompt} 
                            onChange={(e) => setPrompt(e.target.value)} 
                        />
                        <button onClick={handleImageGeneration}>Generate Image</button>
                        <div className="debug-info">
                            <p>Debugging Info for Image Generation:</p>
                            <p>Make sure RapidAPI endpoint is accessible and your key is valid.</p>
                            <p>The image generation might fail if the RapidAPI service is down or has usage limits.</p>
                        </div>
                    </>
                )}
                
                <input type="file" onChange={handleImageUpload} />
                {image && <img src={image} alt="Uploaded" className="mini-image" />}
                {loading && <div className="loader-container"><Oval color="#00BFFF" height={40} width={40} /></div>}
                
                <div className="response-container">
                    <h3>Response {
                        generationMode === 'text' && (
                            apiProvider === 'anthropic' ? ' from Claude' : 
                            apiProvider === 'groq' ? ' from Groq' :
                            apiProvider === 'google' ? ' from Gemini' :
                            apiProvider === 'huggingface' ? ' from HuggingFace' :
                            apiProvider === 'langchain' ? ' from LangChain + Gemini' :
                            apiProvider === 'langgraph' ? ' from LangGraph + Gemini' : ' from OpenAI'
                        )
                    }</h3>
                    <button className="copy-response-button" onClick={copyResponse}>Copy Response</button>
                    
                    {/* Embedded YouTube video player - This now properly handles the video embedding */}
                    {embeddedVideo && (
                        <div className="embedded-video-container">
                            <VideoPlayer embedHtml={embeddedVideo.embedHtml} />
                        </div>
                    )}
                    
                    {generatedImage && generationMode === 'image' ? (
                        <div className="generated-image-container">
                            <img 
                                src={`data:image/jpeg;base64,${generatedImage}`} 
                                alt="Generated" 
                                className="generated-image" 
                            />
                        </div>
                    ) : (
                        <ReactMarkdown components={components} children={response} />
                    )}
                </div>
            </div>
            <div className="conversations-container">
                <h2>Conversations</h2>
                <button onClick={() => setShowConversations(!showConversations)}>
                    {showConversations ? 'Hide Conversations' : 'Show Conversations'}
                </button>
                <button className="delete-all-button" onClick={deleteAllConversations}>Delete All Conversations</button>
                {showConversations && (
                    <ul>
                        {conversations.map((conv) => (
                            <li key={conv.id} className={theme}>
                                <strong>{formatTimestamp(conv.timestamp)}</strong>:
                                
                                {/* Handle different conversation content types */}
                                {conv.image_data ? (
                                    <div className="conversation-image-container">
                                        <p>{conv.conversation}</p>
                                        <img 
                                            src={`data:image/jpeg;base64,${conv.image_data}`} 
                                            alt="Generated" 
                                            className="conversation-image" 
                                        />
                                    </div>
                                ) : conv.has_video || conv.api_provider === 'youtube' || conv.video_id ? (
                                    <div className="conversation-video-container">
                                        <p>{conv.conversation}</p>
                                        <button 
                                            className="play-video-button"
                                            onClick={() => playVideoFromHistory(conv.video_id || conv.video_db_id)}
                                        >
                                            <i className="fas fa-play"></i> Play Video
                                        </button>
                                    </div>
                                ) : (
                                    <ReactMarkdown components={components} children={conv.conversation} />
                                )}
                                
                                <div className="model-info">
                                    <em>Model: {conv.model || "Unknown"}</em>
                                    {conv.temperature && <em> • Temperature: {conv.temperature}</em>}
                                    {conv.api_provider && <em> • Provider: {conv.api_provider}</em>}
                                </div>
                                <button className="delete-button" onClick={() => deleteConversation(conv.id)}>Delete</button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}

export default App;