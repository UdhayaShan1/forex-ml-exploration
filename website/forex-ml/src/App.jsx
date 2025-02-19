import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import SideBar from "./components/SideBar";
import LogisticRegression from "./pages/LogisticRegression";
import Misleading from "./pages/Misleading";
import Home from "./pages/Home";


function App() {
  return (
    <Router>
      <Header />

      <div className="flex">
        <SideBar />

        <div className="p-6" style={{ marginLeft: "300px", width: "calc(100% - 260px)" }}>
          <Routes>
            <Route path="/" element={<Home />} />
          </Routes>

          <Routes>
            <Route path="/logistic" element={<LogisticRegression />} />
          </Routes>
          

          <Routes>
            <Route path="/misleading" element={<Misleading />} />
          </Routes>
        </div>
      </div>

      <Footer />
    </Router>
  );
}

export default App;
