import {html, PolymerElement} from '@polymer/polymer/polymer-element.js';
import '@polymer/paper-button/paper-button.js';

/**
 * @customElement
 * @polymer
 */
class LoginButton extends PolymerElement {
    
  static get template() {
    return html`
      <style>
        :host {
          display: block;
        }
        paper-button {
            background-color:white;
            color: black;
            margin:1rem;
          }
      </style>
      <paper-button raised">Entrar</paper-button>

    `;
  }
  static get properties() {
    return {
       buttonVal:{
        type:String,
      },
    };
  }
}

window.customElements.define('login-button', LoginButton);
