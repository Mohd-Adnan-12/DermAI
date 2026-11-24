
-- DermAI database migration

-- 1. Clinician approval
ALTER TABLE users
ADD COLUMN IF NOT EXISTS is_approved BOOLEAN NOT NULL DEFAULT TRUE;

-- 2. Patient profiles
CREATE TABLE IF NOT EXISTS patient_profiles (
    patient_id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL
        REFERENCES users(user_id) ON DELETE CASCADE,
    patient_name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 1 AND 120),
    gender VARCHAR(30),
    medical_history TEXT,
    previous_skin_treatments TEXT,
    current_medications TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Case information
ALTER TABLE cases
ADD COLUMN IF NOT EXISTS affected_body_part VARCHAR(100);

ALTER TABLE cases
ADD COLUMN IF NOT EXISTS condition_duration VARCHAR(100);

ALTER TABLE cases
ADD COLUMN IF NOT EXISTS symptoms JSONB DEFAULT '[]'::jsonb;
