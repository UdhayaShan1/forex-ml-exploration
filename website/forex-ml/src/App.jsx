import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import SideBar from "./components/SideBar";
import Classification from "./pages/Classification";
import Misleading from "./pages/Misleading";
import Logistic from "./pages/Logistic";
import LogisticImplementation from "./pages/LogisticImplementation";
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
            <Route path="/logistic" element={<Logistic />}></Route>
            <Route path="/logisticimpl" element={<LogisticImplementation></LogisticImplementation>}></Route>
          </Routes>
        </div>
      </div>

      <Footer />
    </Router>
  );
}

export default App;
