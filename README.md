# NammaRide - Integrated Ride-Sharing Platform

## Overview

NammaRide is a comprehensive ride-sharing platform designed to serve both drivers and riders with innovative features that enhance the user experience, safety, and efficiency. The platform consists of three main components:

1. **Driver Interface**
2. **Rider Interface**
3. **Supply-Demand Backend**

## Components

### Driver Interface

The Driver Interface is designed to optimize driver experience and earnings through intelligent features:

- **Driver Preferred Location**: Allows drivers to set preferred operational areas for more targeted ride opportunities
- **Demand Clustering**: Uses advanced clustering algorithms to identify and highlight high-demand areas in real-time
- **Smart Event Analysis**: Leverages Event API and OpenAI integration to predict demand surges around events and gatherings
- **Gamification Features**:
  - Interactive leaderboards to foster healthy competition among drivers
  - Comprehensive earnings overview with analytics and trends
  - Achievement-based incentives to encourage platform engagement

### Rider Interface

The Rider Interface focuses on convenience, safety, and an integrated travel experience:

- **NammaCoins**: Loyalty program that rewards riders with coins redeemable across partner applications and services
- **NammaPink**: Dedicated service for women passengers with verified women drivers, especially for night travel, enhancing safety and peace of mind
- **Metro Yatri Integration**: Seamless connection with metro services for integrated multi-modal transportation
- **Language Localization**: Real-time translation services powered by Sarvam API to break language barriers
- **Pre-booking System**: Allows riders to schedule rides in advance for better planning

### Supply-Demand Backend

The Supply-Demand Backend is the intelligent core of the platform that balances driver availability with rider demand:

- **Predictive Demand Modeling**: Utilizes historical data and machine learning to forecast demand patterns
- **Dynamic Pricing Algorithm**: Automatically adjusts prices based on supply-demand gaps
- **Driver Allocation Optimization**: Efficiently matches drivers to riders considering multiple factors including distance, driver preferences, and rider needs
- **Real-time Analytics Dashboard**: Provides system-wide visibility for operational monitoring
- **API Integration Layer**: Connects with external services including payment gateways, mapping services, and partner applications

## Getting Started

### Prerequisites

- Node.js (v18.0.0 or higher)
- MongoDB (v6.0 or higher)
- Redis (v7.0 or higher)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/namma-ride.git
cd namma-ride
```

2. Install dependencies for all components:
```bash
# Install backend dependencies
cd backend
npm install

# Install driver interface dependencies
cd ../driver-interface
npm install

# Install rider interface dependencies
cd ../rider-interface
npm install
```

3. Set up environment variables:
```bash
# Copy example environment files
cp backend/.env.example backend/.env
cp driver-interface/.env.example driver-interface/.env
cp rider-interface/.env.example rider-interface/.env
```

4. Configure the following environment variables in each .env file:
- `MONGODB_URI`: Your MongoDB connection string
- `REDIS_URL`: Redis connection URL
- `JWT_SECRET`: Secret key for JWT authentication
- `OPENAI_API_KEY`: Your OpenAI API key
- `SARVAM_API_KEY`: Your Sarvam API key for translation services
- `EVENT_API_KEY`: API key for event integration

5. Start the development servers:
```bash
# Start backend server
cd backend
npm run dev

# Start driver interface
cd ../driver-interface
npm run dev

# Start rider interface
cd ../rider-interface
npm run dev
```

## Technology Stack

### Backend
- **Runtime**: Node.js with Express.js
- **Database**: MongoDB with Mongoose ODM
- **Caching**: Redis
- **Authentication**: JWT
- **API Documentation**: Swagger/OpenAPI
- **Testing**: Jest with Supertest

### Frontend (Driver & Rider Interfaces)
- **Framework**: Next.js 14 with React
- **State Management**: Redux Toolkit
- **Styling**: Tailwind CSS
- **Maps Integration**: Mapbox GL JS
- **Real-time Updates**: Socket.IO
- **UI Components**: Shadcn UI
- **Forms**: React Hook Form with Zod validation

### DevOps & Infrastructure
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud Platform**: AWS
- **Monitoring**: Grafana & Prometheus
- **Logging**: ELK Stack

### AI & ML Components
- **Event Analysis**: OpenAI GPT-4
- **Demand Prediction**: TensorFlow
- **Translation**: Sarvam API
- **Clustering**: scikit-learn

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MIT License

Copyright (c) 2024 NammaRide

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
