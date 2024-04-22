# Use Node.js as base image
FROM node:latest

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json files to the working directory
COPY package.json package-lock.json /app/

# Install dependencies
RUN npm install

# Copy the rest of the application code to the working directory
COPY . /app

# Expose port 3000
EXPOSE 3000

# Command to run the application
CMD ["npm", "start"]
