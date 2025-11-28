#!/bin/bash

# Arthya Backend Quick Start Script

echo "🚀 Starting Arthya Backend..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please update with your MongoDB URI if needed."
fi

# Check if MongoDB is running (optional)
echo "📊 Make sure MongoDB is running..."
echo ""

# Build the project
echo "🔨 Building TypeScript..."
npm run build

echo ""
echo "✅ Build complete!"
echo ""
echo "To start the server, run:"
echo "  npm run dev    (for development with hot reload)"
echo "  npm start      (for production)"
echo ""
