import { Link } from "react-router-dom";
import "./SideBar.css";

function SideBar({ isOpen }) {
  return (
    <div className={`sidebar ${isOpen ? "open" : ""}`}>
      <h2 className="sidebar-title">Machine Learning (For)ex</h2>
      <p className="sidebar-text">Explore the different sections here!</p>
      <ul className="sidebar-list">
        <li><Link to="/" className="sidebar-link">Home</Link></li>
        <p>The Motivation</p>
        <li><Link to="/misleading" className="sidebar-link">Misleading Regression</Link></li>
        <li><Link to="/classification" className="sidebar-link">Classification Instead?</Link></li>
        <p>Models</p>
        <li><Link to="/logistic" className="sidebar-link">Logistic Regression</Link></li>
      </ul>
    </div>
  );
}

export default SideBar;
