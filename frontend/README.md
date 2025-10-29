# Frontend — AccentDetectApp (Expo + React Native Paper)

This is a minimal Expo React Native app that demonstrates a UI for uploading/recording an audio file and sending it to the backend `/detect-accent` endpoint.

Quick start

1. Install dependencies (make sure you have Node.js and npm installed):

```bash
cd frontend
npm install
```

2. Start the Expo dev server:

```bash
npm run start
```

3. Run on an emulator or physical device. If running on Android emulator, use `10.0.2.2` instead of `localhost` for the backend host.

Configuration

- The frontend client defaults to `http://localhost:8000` for the backend. If you need to change it (e.g. emulator): set the environment variable `ACCENT_API_URL` before running Expo, or edit `src/api/client.ts`.

Generating a typed client

An example script placeholder is present in `package.json` under `generate-client`. To generate a client with `@hey-api/openapi-ts`:

1. Install generator (if you want to use it):

```bash
npm install -D @hey-api/openapi-ts
# or globally: npm i -g @hey-api/openapi-ts
```

2. Run generator pointing to the backend `openapi.json`:

```bash
# from frontend/ folder
npx @hey-api/openapi-ts --input ../backend/openapi.json --output src/api/generated --client fetch
```

This repo includes a small hand-written `src/api/client.ts` as a placeholder so you can test the flow immediately.
