import {html, PolymerElement} from '@polymer/polymer/polymer-element.js';

/**
 * @customElement
 * @polymer
 */
class BbvaGraphicsApp extends PolymerElement {
  static get template() {
    return html`
      <style>
        :host {
          display: block;
        }
      </style>
      <iframe width="350" height="430" allow="microphone;" 
      src="https://console.dialogflow.com/api-client/demo/embedded/76dee451-46ba-4568-a2ce-c7bc06faa5f9"></iframe>
    `;
  }
  static get properties() {
    return {
      prop1: {
        type: String,
        value: 'bbva-graphics-app'
      }
    };
  }
}

window.customElements.define('bbva-graphics-app', BbvaGraphicsApp);
