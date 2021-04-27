FROM node:14-alpine

WORKDIR /opt/pepper-web-frontend/

# install app dependencies
COPY package.json ./
COPY package-lock.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]

