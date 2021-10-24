import {html, PolymerElement} from '@polymer/polymer/polymer-element.js';
import '@polymer/iron-input/iron-input.js';

/**
 * @customElement
 * @polymer
 */
class MybbvaAppApp extends PolymerElement {
  static get template() {
    return html`
      <style>
        :host {
          display: block;
        }
        iron-input{
          font-size:2rem;
          width:100%;
        }
        
      </style>

      <label>
      <iron-input>
        <input placeholder="[[prop1]]" type=[[type]]>
      </iron-input>
    `;
  }
  static get properties() {
    return {
      prop1:{
        type:String,
      },
      type:{
        type:String
      }
    };
  }
}

window.customElements.define('mybbva-app-app', MybbvaAppApp);
