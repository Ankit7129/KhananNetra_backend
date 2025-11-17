FROM node:22-slim

WORKDIR /app

# Copy only package.json and package-lock.json first
COPY package.json package-lock.json ./

RUN npm ci --omit=dev

# Now copy the rest of the app
COPY . .

EXPOSE 8080

CMD ["npm", "start"]

