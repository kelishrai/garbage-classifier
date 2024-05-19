import axios, { AxiosInstance, CreateAxiosDefaults } from "axios";

type RequestType = {
  url: string;
  body?: any;
};

export class Api {
  private instance;
  private static config: CreateAxiosDefaults<any> = {
    // baseURL: "https://dummyjson.com",
    baseURL: "http://127.0.0.1:8000/",
  };

  constructor() {
    this.instance = axios.create(Api.config);
  }

  public async request({
    method,
    url,
    body,
  }: { method: "GET" | "POST" } & RequestType) {
    switch (method) {
      case "POST":
        return this.post({ url, body });
      case "GET":
        return this.get({ url });
    }
  }

  private async post({ url, body }: RequestType): Promise<any> {
    return this.instance.post(url, body);
  }

  private async get({ url }: RequestType): Promise<any> {
    return this.instance.get(url);
  }
}
