// src/api/client.ts
import { client } from "../client/client.gen"; // được tạo bởi openapi-ts
import * as SecureStore from "expo-secure-store";

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || "http://192.168.1.10:8000/api/v1";

client.setConfig({
  baseUrl: API_BASE_URL,
});

// 🧩 Thêm interceptor tự động chèn token
client.interceptors.request.use(async (request) => {
  const token = await SecureStore.getItemAsync("access_token");
  if (token) {
    request.headers.set("Authorization", `Bearer ${token}`);
  }
  return request;
});

// 🧩 (Tùy chọn) xử lý lỗi 401
client.interceptors.error.use(async (error, response) => {
  if (response?.status === 401) {
    console.warn("Token hết hạn hoặc không hợp lệ.");
    await SecureStore.deleteItemAsync("access_token");
  }
  return Promise.reject(error);
});

export { client };
