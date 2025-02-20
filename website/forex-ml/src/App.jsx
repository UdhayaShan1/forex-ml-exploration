import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import SideBar from "./components/SideBar";
import Classification from "./pages/Classification";
import Misleading from "./pages/Misleading";
import Home from "./pages/Home";
import { useState } from "react";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <Router>

      <button 
        className="menu-button"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        ☰
      </button>

      <div className="app-container">
        <SideBar isOpen={sidebarOpen} />
        
        <div className="content-container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/classification" element={<Classification />} />
            <Route path="/misleading" element={<Misleading />} />
          </Routes>
        </div>
      </div>

      <Footer />
    </Router>
  );
}

export default App;
