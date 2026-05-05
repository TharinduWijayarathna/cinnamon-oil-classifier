FROM node:20-alpine AS build

WORKDIR /ui

COPY ui/package*.json ./
RUN npm ci

COPY ui ./

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 80
