-- DermAI database compatibility migration
-- Run from the DermAI project folder using:
-- psql -d dermai_db -f dermai_setup.sql

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_approved BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS patient_profiles (
    patient_id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    patient_name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 1 AND 120),
    gender VARCHAR(30),
    weight_kg NUMERIC(5,2),
    medical_history TEXT,
    previous_skin_treatments TEXT,
    current_medications TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE patient_profiles
    ADD COLUMN IF NOT EXISTS medical_history TEXT;
ALTER TABLE patient_profiles
    ADD COLUMN IF NOT EXISTS previous_skin_treatments TEXT;
ALTER TABLE patient_profiles
    ADD COLUMN IF NOT EXISTS current_medications TEXT;
ALTER TABLE patient_profiles
    ADD COLUMN IF NOT EXISTS weight_kg NUMERIC(5,2);

ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS affected_body_part VARCHAR(100);
ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS condition_duration VARCHAR(100);
ALTER TABLE cases
    ADD COLUMN IF NOT EXISTS symptoms JSONB DEFAULT '[]'::jsonb;

-- Existing account names are used as patient profile names when profiles
-- already exist. New registrations create the profile automatically.
