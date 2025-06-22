import React, { useState, useEffect, useMemo } from 'react';
import './App.css';

interface TokenShapValue {
  token: string;
  value: number;
}

interface ArticleShapData {
  title: string;
  abstract: string;
  verdict: string;
  explained_title_parts: TokenShapValue[];
  explained_abstract_parts: TokenShapValue[];
  top_words: TokenShapValue[];
}

interface TopicDataResponse {
  topic_name: string;
  articles: ArticleShapData[];
}

const API_BASE_URL = 'http://localhost:8000';

const topicDescriptions: Record<string, string> = {
  CD011975: "Title: First trimester serum tests for Down's syndrome screening ",
  CD012599: "Title: First and second trimester serum tests with and without first trimester ultrasound tests for Down's syndrome screening ",
  CD010213: "Title: Imaging modalities for characterising focal pancreatic lesions "
};

function App() {
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [currentTopicData, setCurrentTopicData] = useState<TopicDataResponse | null>(null);
  const [selectedArticleIndex, setSelectedArticleIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [userVerdicts, setUserVerdicts] = useState<Record<number, string>>({});

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
        const rawTopics: string[] = await response.json();
        const baseTopicSet = new Set<string>();
        rawTopics.forEach(t => baseTopicSet.add(t.split('_')[0]));
        setTopics(Array.from(baseTopicSet));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
      }
    };
    fetchTopics();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!currentTopicData || selectedArticleIndex === null) return;
      if (event.key === 'ArrowLeft' && selectedArticleIndex > 0) {
        setSelectedArticleIndex(prev => prev !== null ? prev - 1 : prev);
      }
      if (
        event.key === 'ArrowRight' &&
        selectedArticleIndex < currentTopicData.articles.length - 1
      ) {
        setSelectedArticleIndex(prev => prev !== null ? prev + 1 : prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedArticleIndex, currentTopicData]);

  useEffect(() => {
    if (selectedTopic) {
      const fetchShapData = async () => {
        setIsLoading(true);
        setError(null);
        setCurrentTopicData(null);
        setSelectedArticleIndex(null);
        setUserVerdicts({});
        try {
          const response = await fetch(`${API_BASE_URL}/shap_data/${selectedTopic}`);
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(`Failed to fetch SHAP data for topic ${selectedTopic}: ${errorData.detail || response.statusText}`);
          }
          const data: TopicDataResponse = await response.json();
          setCurrentTopicData(data);
          if (data.articles && data.articles.length > 0) {
            setSelectedArticleIndex(0);
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

  const handleTopicChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const baseId = event.target.value.split('_')[0];
    setSelectedTopic(baseId || null);
  };

  const handleArticleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(event.target.value, 10);
    setSelectedArticleIndex(isNaN(index) ? null : index);
  };

  const handleUserVerdict = (verdict: string) => {
    if (selectedArticleIndex !== null) {
      setUserVerdicts(prev => ({ ...prev, [selectedArticleIndex]: verdict }));
    }
  };

  const getShapColor = (value: number, maxValue: number = 0.5): string => {
    if (value === 0) return 'transparent';
    const intensity = Math.min(Math.abs(value) / maxValue, 1);
    const alpha = intensity * 0.6 + 0.1;
    if (value > 0.001) return `rgba(0, 128, 0, ${alpha})`;
    if (value < -0.001) return `rgba(255, 0, 0, ${alpha})`;
    return 'transparent';
  };

  const renderShapText = (parts: TokenShapValue[], defaultText: string) => {
    if (!parts || parts.length === 0) {
      return <p><em>{defaultText}</em></p>;
    }
    return parts.map((part, index) => {
      const leadingSpace = index > 0 && part.token.match(/^\w/) ? ' ' : '';
      return (
        <React.Fragment key={index}>
          {leadingSpace}
          <span
            style={{
              backgroundColor: getShapColor(part.value),
              padding: '1px 0',
              borderRadius: '3px'
            }}
            title={`SHAP value: ${part.value.toFixed(5)}`}
          >
            {part.token}
          </span>
        </React.Fragment>
      );
    });
  };

  const currentUserVerdict = selectedArticleIndex !== null ? userVerdicts[selectedArticleIndex] : null;

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

          {selectedTopic && topicDescriptions[selectedTopic] && (
            <div className="topic-description">
              <h2>{topicDescriptions[selectedTopic]}</h2>
            </div>
          )}

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
            <div className="shap-text-display">
              <div className="article-nav-buttons">
                <button
                  onClick={() => setSelectedArticleIndex((prev) => (prev !== null && prev > 0 ? prev - 1 : prev))}
                  disabled={selectedArticleIndex === null || selectedArticleIndex <= 0}
                >
                  ← Previous
                </button>
                <button
                  onClick={() => setSelectedArticleIndex((prev) =>
                    prev !== null && currentTopicData && prev < currentTopicData.articles.length - 1 ? prev + 1 : prev
                  )}
                  disabled={
                    selectedArticleIndex === null ||
                    !currentTopicData ||
                    selectedArticleIndex >= currentTopicData.articles.length - 1
                  }
                >
                  Next →
                </button>
              </div>
              <h2 className="article-title">
                {renderShapText(currentArticle.explained_title_parts, currentArticle.title || 'Title not available.')}
              </h2>
              <div className="abstract-section">
                <strong>Abstract:</strong>
                <p>
                  {renderShapText(currentArticle.explained_abstract_parts, currentArticle.abstract || 'Abstract not available.')}
                </p>
              </div>
            </div>

            <div className="verdict-section">
              <p className="verdict">
                <strong>Model Suggestion:</strong>{' '}
                <span className={currentArticle.verdict?.toLowerCase()}>{currentArticle.verdict}</span>
              </p>

              <p className="user-verdict">
                <strong>Your Verdict:</strong>{' '}
                {currentUserVerdict ? (
                  <span className={currentUserVerdict.toLowerCase()}>{currentUserVerdict}</span>
                ) : (
                  <em>Not selected</em>
                )}
              </p>

              <div className="verdict-buttons">
                <button onClick={() => handleUserVerdict('Relevant')} className="verdict-btn relevant">
                  Mark as Relevant
                </button>
                <button onClick={() => handleUserVerdict('Irrelevant')} className="verdict-btn irrelevant">
                  Mark as Irrelevant
                </button>
              </div>
            </div>

            <div className="top-words">
              <h3>Top 5 Words Contributing to '{currentArticle.verdict}' Verdict</h3>
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
