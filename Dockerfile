FROM node:20-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package*.json ./
RUN npm install --omit=dev

# Copy the rest of the app
COPY . .

# Back4app injects PORT at runtime; server.js already reads process.env.PORT
EXPOSE 3000

CMD ["node", "server.js"]
