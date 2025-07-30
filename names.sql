CREATE TABLE predictions (
    id INT not null AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    predicted_gender VARCHAR(10),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
