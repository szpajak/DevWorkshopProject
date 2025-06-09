import React, { useState, useEffect, useMemo } from 'react';
import './App.css'; 

interface TokenShapValue {
  token: string;
  value: number; // SHAP value for the "Relevant" class
}

interface ArticleShapData {
  title: string;
  abstract: string;
  verdict: string;
  explained_text_parts: TokenShapValue[];
  top_words: TokenShapValue[];
}

interface TopicDataResponse {
  topic_name: string;
  articles: ArticleShapData[];
}

const API_BASE_URL = 'http://localhost:8000'; // Backend API URL

function App() {
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [currentTopicData, setCurrentTopicData] = useState<TopicDataResponse | null>(null);
  const [selectedArticleIndex, setSelectedArticleIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTopics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/topics`);
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(`Failed to fetch topics: ${errorData.detail || response.statusText}`);
        }
        const data: string[] = await response.json();
        setTopics(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
      }
    };
    fetchTopics();
  }, []);

  useEffect(() => {
    if (selectedTopic) {
      const fetchShapData = async () => {
        setIsLoading(true);
        setError(null);
        setCurrentTopicData(null);
        setSelectedArticleIndex(null);
        try {
          const response = await fetch(`${API_BASE_URL}/shap_data/${selectedTopic}`);
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(`Failed to fetch SHAP data for topic ${selectedTopic}: ${errorData.detail || response.statusText}`);
          }
          const data: TopicDataResponse = await response.json();
          setCurrentTopicData(data);
          if (data.articles && data.articles.length > 0) {
            setSelectedArticleIndex(0); // Auto-select first article
          } else {
            setError(`No articles found for topic ${selectedTopic}.`);
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
          setCurrentTopicData(null);
        } finally {
          setIsLoading(false);
        }
      };
      fetchShapData();
    }
  }, [selectedTopic]);

  const currentArticle = useMemo(() => {
    if (currentTopicData && selectedArticleIndex !== null && currentTopicData.articles[selectedArticleIndex]) {
      return currentTopicData.articles[selectedArticleIndex];
    }
    return null;
  }, [currentTopicData, selectedArticleIndex]);

  // DEBUGGING: Log currentArticle and its relevant parts when it changes
  useEffect(() => {
    if (currentArticle) {
      console.log("Current Article Data Received by Frontend:", currentArticle);
      console.log("Explained Text Parts:", currentArticle.explained_text_parts);
      console.log("Top Words:", currentArticle.top_words);
    }
  }, [currentArticle]);


  const handleTopicChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTopic(event.target.value || null);
  };

  const handleArticleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(event.target.value, 10);
    setSelectedArticleIndex(isNaN(index) ? null : index);
  };

  const getShapColor = (value: number, maxValue: number = 0.5): string => {
    if (value === 0) return 'transparent';
    const intensity = Math.min(Math.abs(value) / maxValue, 1); 
    const alpha = intensity * 0.6 + 0.1; 
    if (value > 0.001) {
      return `rgba(0, 128, 0, ${alpha})`; 
    } else if (value < -0.001) {
      return `rgba(255, 0, 0, ${alpha})`; 
    }
    return 'transparent';
  };

  const renderShapText = (parts: TokenShapValue[]) => {
    if (!parts || parts.length === 0) {
      return <p><em>SHAP text data is not available for highlighting.</em></p>;
    }
    // const maxAbsValue = parts.reduce((max, p) => Math.max(max, Math.abs(p.value)), 0.00001);

    return parts.map((part, index) => {
      let tokenText = part.token;
      let addSpaceBefore = index > 0; 
      if (tokenText.startsWith("##")) {
        tokenText = tokenText.substring(2); 
        addSpaceBefore = false; 
      } else if (index > 0) {
        const prevToken = parts[index-1].token;
        if (tokenText.match(/^[,.!?':;\]\)\}\-\–\—]$/)) { 
            addSpaceBefore = false;
        } else if (prevToken.match(/[\(\[\{\"\']$/)) { 
            addSpaceBefore = false;
        }
      }
      return (
        <React.Fragment key={index}>
          {addSpaceBefore && ' '}
          <span 
            style={{ 
              backgroundColor: getShapColor(part.value /*, maxAbsValue */), 
              padding: '1px 0', 
              margin: '0',       
              borderRadius: '3px' 
            }}
            title={`SHAP value: ${part.value.toFixed(5)}`} 
          >
            {tokenText}
          </span>
        </React.Fragment>
      );
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>SHAP Value Interpretation</h1>
      </header>
      <main className="App-main">
        {isLoading && <p className="loading-message">Loading...</p>}
        {error && <p className="error-message">Error: {error}</p>}

        <div className="controls">
          <div className="control-group">
            <label htmlFor="topic-select">Choose a Topic:</label>
            <select id="topic-select" onChange={handleTopicChange} value={selectedTopic || ''} disabled={isLoading || topics.length === 0}>
              <option value="" disabled={!!selectedTopic}>-- Select Topic --</option>
              {topics.map(topic => (
                <option key={topic} value={topic}>{topic}</option>
              ))}
            </select>
          </div>

          {selectedTopic && currentTopicData && (
            <div className="control-group">
              <label htmlFor="article-select">Choose an Article:</label>
              <select 
                id="article-select" 
                onChange={handleArticleChange} 
                value={selectedArticleIndex !== null ? selectedArticleIndex : ''} 
                disabled={isLoading || !currentTopicData.articles.length}
              >
                <option value="" disabled={selectedArticleIndex !== null}>-- Select Article --</option>
                {currentTopicData.articles.map((article, index) => (
                  <option key={index} value={index}>
                    {article.title ? article.title.substring(0, 70) + (article.title.length > 70 ? '...' : '') : `Article ${index + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {currentArticle && (
          <div className="article-details">
            {/* Integrated SHAP-highlighted text (replaces separate title and abstract) */}
            <div className="shap-text-display main-highlighted-text">
              {currentArticle.explained_text_parts && currentArticle.explained_text_parts.length > 0 ?
                renderShapText(currentArticle.explained_text_parts)
                :
                ( /* Fallback to plain title and abstract if SHAP parts are missing */
                  <>
                    <h2>{currentArticle.title}</h2> 
                    {currentArticle.abstract && (
                      <div className="abstract-section">
                        <strong>Abstract:</strong>
                        <p>{currentArticle.abstract}</p>
                      </div>
                    )}
                    <p><em>SHAP highlighting data is not available for this article.</em></p>
                  </>
                )
              }
            </div>

            <p className="verdict"><strong>Verdict:</strong> <span className={currentArticle.verdict?.toLowerCase()}>{currentArticle.verdict}</span></p>
            
            <div className="top-words">
              <h3>Top 5 Contributing Words (for Relevancy)</h3>
              {currentArticle.top_words && currentArticle.top_words.length > 0 ? (
                <ul>
                  {currentArticle.top_words.map((word, index) => (
                    <li key={index} style={{ color: word.value > 0 ? 'darkgreen' : 'darkred' }}>
                      "{word.token}": {word.value.toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p><em>Top contributing words data is not available for this article.</em></p>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
