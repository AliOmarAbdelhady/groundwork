import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import App from "./App";
import ChatPage from "./pages/ChatPage";
import ExplorePage from "./pages/ExplorePage";
import EvalPage from "./pages/EvalPage";
import InsightsPage from "./pages/InsightsPage";
import "./index.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <ChatPage /> },
      { path: "explore", element: <ExplorePage /> },
      { path: "evaluation", element: <EvalPage /> },
      { path: "insights", element: <InsightsPage /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
