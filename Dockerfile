FROM node:20-alpine
WORKDIR /app
COPY src ./src
COPY test ./test
CMD ["node", "--test", "test"]
