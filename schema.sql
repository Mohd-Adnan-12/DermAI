
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cases (
    case_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(30) DEFAULT 'pending'
);

CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(case_id),
    image_path TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(image_id),
    predicted_class VARCHAR(10) NOT NULL,
    confidence NUMERIC(5,4),
    model_version VARCHAR(50),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prediction_history (
    history_id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL
        REFERENCES predictions(prediction_id),
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
