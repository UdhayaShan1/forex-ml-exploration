import { Link } from "react-router-dom";
import "./SideBar.css";

function SideBar({ isOpen }) {
  return (
    <div className={`sidebar ${isOpen ? "open" : ""}`}>
      <h2 className="sidebar-title">Machine Learning (For)ex</h2>
      <p className="sidebar-text">Explore the different sections here!</p>
      <ul className="sidebar-list">
        <li><Link to="/" className="sidebar-link">Home</Link></li>
        <p className="sidebar-seperator">The Motivation</p>
        <li><Link to="/misleading" className="sidebar-link">Misleading Regression</Link></li>
        <li><Link to="/classification" className="sidebar-link">Classification And Metrics</Link></li>
        <p className="sidebar-seperator">Simple Models</p>
        <li><Link to="/logistic" className="sidebar-link">Intro to Logistic Regression</Link></li>
        <li><Link to="/logisticimpl" className="sidebar-link">Logistic Regression Analysis</Link></li>
      </ul>
    </div>
  );
}

export default SideBar;
